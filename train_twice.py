"""二次训练: 在 1-1 的 best_model 基础上继续训 1-2。

    python train_twice.py                      # 阶段A: 锁关 1-2, 从 best_model 热启动, 500 万步
    python train_twice.py --eval --episodes 5  # 看看阶段A的模型在 1-2 上跑得怎样
    python train_twice.py --stage full         # 阶段B: 回完整游戏巩固 1-1 -> 1-2

不依赖 my_obs.py 的改动, 也不改动 my_wrapper.py / train_model.py / test_model.py。
"""
import argparse
import time
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage,
)

from my_wrapper import MarioRewardWrapper, SkipFrameWrapper

# ---- 两个阶段的默认参数 ----
STAGES = {
    # 锁关 1-2: 每局直接从 1-2 开头开始, 样本效率最高
    "1-2": dict(env_id="SuperMarioBros-1-2-v2", single_stage=True,
                resume="best_model/best_model.zip",      # 能过 1-1 的那份权重
                timesteps=5_000_000),
    # 完整游戏: 巩固 1-1, 并学会连起来打
    "full": dict(env_id="SuperMarioBros-v2", single_stage=False,
                 resume="models/1-2/best_model.zip",     # 阶段 A 的产物
                 timesteps=2_000_000),
}


class DeathPenaltyOnDone(gym.Wrapper):
    """锁关环境在 _is_dying 那一帧就 done, 此时 life 还没减,
    MarioRewardWrapper 里 "life 下降" 的死亡判据永远不触发。

    这里在它外层补一次 "本局结束 且 没举旗 == 死亡" 的扣分。
    只在锁关环境挂载: 完整游戏模式下 done 只在 life==0xFF 时成立,
    挂上去会和 MarioRewardWrapper 重复扣分。
    """

    def __init__(self, env, penalty=15.0):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and not info["flag_get"]:
            reward -= self.penalty
        return obs, reward, done, info


def make_env(rank, env_id, single_stage, time_penalty=0.01, death_penalty=15.0):
    """复刻 my_obs.make_env 的包装链, 只是把关卡和奖励参数变成可配的。

    rank 目前只用于区分进程, 不影响行为。
    """
    # 多进程需要 "造环境的函数", 不能直接传环境
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrameWrapper(env, skip=4)
        env = MarioRewardWrapper(env, time_penalty=time_penalty,
                                 death_penalty=death_penalty)
        if single_stage:
            env = DeathPenaltyOnDone(env, penalty=death_penalty)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        return env

    return _init


def build_vec_env(n, env_id, single_stage, time_penalty, death_penalty, offset=0):
    vec_env = SubprocVecEnv([
        make_env(offset + i, env_id, single_stage, time_penalty, death_penalty)
        for i in range(n)
    ])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="last")
    vec_env = VecMonitor(vec_env)
    return vec_env


def build_model(args, vec_env):
    """有 --resume 就热启动, 否则从头训。"""
    tb_log = f"logs/{args.stage}"

    if args.resume is None:
        model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log=tb_log,
                    learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=4,
                    ent_coef=args.ent_coef, gamma=args.gamma)
        print("[起点] 从零开始训练")
        return model

    if not Path(args.resume).exists():
        raise FileNotFoundError(f"找不到权重文件: {args.resume}")

    model = PPO.load(str(args.resume), env=vec_env, tensorboard_log=tb_log)
    # 载入的 lr_schedule 是上一轮训练留下的闭包, 换成二次训练自己的调度。
    # 注意: 下面的 learn(reset_num_timesteps=True) 会让 progress_remaining 从 1.0
    # 重新开始算, 所以这里 lr_start/lr_end 才是"本次训练"真实的起止学习率。
    model.lr_schedule = get_linear_fn(args.lr_start, args.lr_end, 0.8)
    model.ent_coef = args.ent_coef
    model.gamma = args.gamma
    print(f"[起点] 从 {args.resume} 热启动, 该权重已训练 {model.num_timesteps} 步")
    return model


def train(args):
    stage = STAGES[args.stage]
    run_dir = Path("models") / args.stage
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    n_envs = args.n_envs
    vec_env = build_vec_env(n_envs, stage["env_id"], stage["single_stage"],
                            args.time_penalty, args.death_penalty, offset=0)
    # 评估环境必须是独立的一套, 不能复用训练环境
    eval_env = build_vec_env(n_envs, stage["env_id"], stage["single_stage"],
                             args.time_penalty, args.death_penalty, offset=100)
    # PPO 会自动给训练环境套一层 VecTransposeImage(HWC -> CHW), 评估环境手动跟上,
    # 否则 EvalCallback 会刷 "Training and eval env are not of the same type" 警告。
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_dir),        # -> models/<stage>/best_model.zip
        log_path=str(run_dir),                    # -> models/<stage>/evaluations.npz
        eval_freq=max(100_000 // n_envs, 1),      # 每 10 万 agent step 评估一次
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),      # 每 50 万 agent step 存一份
        save_path=str(run_dir / "checkpoints"),
        name_prefix=args.stage,
    )

    model = build_model(args, vec_env)

    print(f"[环境] {stage['env_id']}  n_envs={n_envs}")
    print(f"[步数] 本次再训 {int(args.timesteps):,} 步")
    print(f"[学习率] {args.lr_start} -> {args.lr_end}")
    print(f"[存档] {run_dir}")

    # reset_num_timesteps=True 让 learning-rate 调度按"本次训练"的进度走
    # (progress_remaining 从 1.0 开始)。若为 False, SB3 会拿累积步数去比 total,
    # 热启动时 progress 一上来就接近 0, lr 会直接掉到 lr_end, 参数形同失效。
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"PPO_{args.stage}",
        reset_num_timesteps=True,
    )
    model.save(str(run_dir / "final"))
    print(f"[完成] 已保存到 {run_dir / 'final.zip'}")


def resolve_eval_model(args, stage):
    """评估读哪份权重: 显式 --resume 优先, 否则优先本阶段产物, 再否则退回起点权重。"""
    if args.resume_explicit:
        return Path(args.resume)
    trained = Path("models") / args.stage / "best_model.zip"
    if trained.exists():
        return trained
    return Path(stage["resume"])


def evaluate(args):
    stage = STAGES[args.stage]
    model_path = resolve_eval_model(args, stage)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {model_path}")

    env = DummyVecEnv([make_env(0, stage["env_id"], stage["single_stage"],
                                args.time_penalty, args.death_penalty)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    model = PPO.load(str(model_path), env=env)
    print(f"[评估] {model_path}  on {stage['env_id']}")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total, max_x, flag = 0.0, 0, False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
            max_x = max(max_x, info[0]["x_pos"])     # DummyVecEnv 的 info 是 list
            flag = flag or bool(info[0]["flag_get"])
            if not args.headless:
                env.render()
                time.sleep(args.delay)
            done = bool(done[0])
        print(f"  第 {ep + 1} 局: 最远 x={max_x}  reward={total:.1f}  "
              f"{'* 举旗通关' if flag else '未通关'}")


def parse_args():
    p = argparse.ArgumentParser(description="在 1-1 权重上二次训练")
    p.add_argument("--stage", choices=list(STAGES), default="1-2",
                   help="1-2 = 锁关训练 1-2; full = 完整游戏巩固")
    p.add_argument("--resume", default=None,
                   help="热启动权重; 默认用 STAGES 里配的那个")
    p.add_argument("--no-resume", action="store_true", help="强制从零开始")
    p.add_argument("--timesteps", type=float, default=None, help="本次训练步数")
    p.add_argument("--n-envs", type=int, default=4)
    # 环境是确定性的 + deterministic=True, 同一策略每局轨迹完全一致,
    # 跑多局只是把同一局重复 N 次(实测 5 局的 std 精确为 0.00)。默认 1 局即可。
    p.add_argument("--n-eval-episodes", type=int, default=1)
    p.add_argument("--lr-start", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--ent-coef", type=float, default=0.015)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--time-penalty", type=float, default=0.03)
    p.add_argument("--death-penalty", type=float, default=15.0)
    p.add_argument("--eval", action="store_true", help="只评估, 不训练")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--headless", action="store_true", help="评估时不渲染")
    p.add_argument("--delay", type=float, default=0.05)
    args = p.parse_args()

    stage = STAGES[args.stage]
    args.resume_explicit = args.resume is not None
    if args.resume is None and not args.no_resume:
        args.resume = stage["resume"]
    if args.timesteps is None:
        args.timesteps = stage["timesteps"]
    return args


def main():
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
