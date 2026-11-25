"""二次训练: 在 1-1 的 best_model 基础上继续训 1-2。

    python train_twice.py                      # 阶段A: 锁关 1-2, 从 best_model 热启动, 500 万步
    python train_twice.py --eval --episodes 5  # 看看阶段A的模型在 1-2 上跑得怎样
    python train_twice.py --stage full         # 阶段B: 回完整游戏巩固 1-1 -> 1-2

    # 1-3 是移动平台关, 确定性环境里 PPO 会锁死在固定轨迹上, 要开随机起手:
    python train_twice.py --stage 1-3 --timesteps 5e6 \
        --ent-coef 0.05 --lr-start 3e-4 --time-penalty 0.01 --random-noop 30

不依赖 my_obs.py 的改动, 也不改动 my_wrapper.py / train_model.py / test_model.py。
"""
import argparse
import time
from pathlib import Path

import gym
import gym_super_mario_bros
import numpy as np
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
    "1-3": dict(env_id="SuperMarioBros-1-3-v2", single_stage=True,
                resume="best_model/best_model.zip",     # ← 2M 巩固模型
                timesteps=5_000_000),
    # 完整游戏: 巩固 1-1, 并学会连起来打
    # 2M 步只够巩固两关(1175 步/局), 四关连过单局会到 2500+ 步, 需要 6M 起
    "full": dict(env_id="SuperMarioBros-v2", single_stage=False,
                 resume="models/1-3/best_model.zip",     # 阶段 A 的产物
                 timesteps=6_000_000),
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


class RandomNoopStart(gym.Wrapper):
    """每局开始先随机按 0~max_noop 次 NOOP(什么都不按), 再交给策略。

    为什么需要: 环境是确定性的, PPO 又是 on-policy(没有 replay buffer),
    策略一旦收敛到某条固定路线, 每局轨迹几乎完全相同 -> 回报方差趋零 ->
    优势恒为 0 -> 梯度消失。1-3 前两轮就是这么冻住的: ep_rew_mean 从 50 万
    步起一动不动, ep_len_mean 恒等于 66.0, approx_kl 衰减到 0.002。
    加大 ent_coef 无效, 因为它只让策略"换个死法", 没采到任何新状态。

    对 1-3 还有一层: 升降台按固定周期跑, 起手帧固定 => 每一局都是同一个相位
    => 只能学到"我第 N 步跳"这种记忆型解, 进了完整游戏前几步一偏就废。
    随机 NOOP 平移了起手相位, 逼策略学成"平台在这个高度该怎么动"。

    用 NOOP 而不是随机动作: NOOP 时 Mario 站着不动, 只有时间在走,
    精确地只平移周期型障碍的相位, 不破坏关卡语义。随机动作会乱跑乱跳,
    采到的多半是"已经掉坑里"的垃圾状态。

    参考 Atari 标准实验协议里的 random no-op starts (Mnih et al., 2015)。
    """

    def __init__(self, env, max_noop=30):
        super().__init__(env)
        self.max_noop = max_noop

    def reset(self):
        obs = self.env.reset()
        for _ in range(np.random.randint(0, self.max_noop + 1)):
            obs, _, done, _ = self.env.step(0)      # 0 = NOOP
            if done:                                # 起手就死, 直接重开
                return self.env.reset()
        return obs


class FrontierStart(gym.Wrapper):
    """第一次跑到 start_x 就把那帧的 NES 状态存成检查点, 之后每局都从那里开始。

    为什么需要: 1-3 的死点在 x≈645 之后(右侧整屏没有静态落脚点, 要等升降台)。
    但这一决策出现在整局第 46 步, 回报要到第 66 步才结算, PPO 的探索够不着 ——
    而且"一路右跑然后死"拿 +59 分, 比"停下等"的 −15 分还高, 送死是局部最优。
    把起点挪到死点前, 关键决策就落到开局前 5 步, 每局都变成一次"在边缘怎么办"的采样。

    思路来自 Go-Explore (Ecoffet et al., Nature 2021) 的 "return then explore"
    和 reverse curriculum generation (Florensa et al., CoRL 2017),
    这里简化成一个固定检查点, 不维护状态档案。

    机制: nes_py 只有一个备份槽, reset() 从它恢复。覆盖它, reset() 就回到检查点。
    """

    def __init__(self, env, start_x=600):
        super().__init__(env)
        self.start_x = start_x
        self._captured = False
        self._last_y = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # 判据: 连续两步 y 不变 = 踩在实地上。
        # 不能用 _y_viewport == 1, 坠落时它也是 1, 存到半空就全废了。
        y = info["y_pos"]
        standing = (y == self._last_y)
        self._last_y = y
        if (not self._captured and standing and not done
                and info["x_pos"] >= self.start_x):
            self.env.unwrapped._backup()      # 覆盖 base env 的备份槽
            self._captured = True
            print(f"[检查点] 已固化在 x={info['x_pos']}, 之后每局从这里开始",
                  flush=True)
        return obs, reward, done, info


def make_env(rank, env_id, single_stage, time_penalty=0.01, death_penalty=15.0,
             random_noop=0, start_x=0):
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
        # 检查点必须在 RandomNoopStart 内侧: NOOP 要作用在检查点上,
        # 这样才能平移升降台的相位(策略此刻就在决策点旁边)
        if single_stage and start_x > 0:
            env = FrontierStart(env, start_x=start_x)
        # 只在训练环境挂: 评估要保持确定性, best_model 才有可比性
        if random_noop > 0:
            env = RandomNoopStart(env, max_noop=random_noop)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        return env

    return _init


def build_vec_env(n, env_id, single_stage, time_penalty, death_penalty,
                  random_noop=0, start_x=0, frontier_prob=0.0, offset=0):
    # nes_py 只有一个备份槽, 抓了检查点就再也回不到关卡开头。
    # 所以只让 frontier_prob 比例的 worker 抓, 其余每局仍从关卡开头跑,
    # 免得采样起点全挤在死点、策略忘了怎么从开头跑过来。
    env_fns = []
    for i in range(n):
        this_start_x = start_x if (start_x > 0 and i / n < frontier_prob) else 0
        env_fns.append(make_env(offset + i, env_id, single_stage, time_penalty,
                                death_penalty, random_noop, this_start_x))
    vec_env = SubprocVecEnv(env_fns)
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
                            args.time_penalty, args.death_penalty,
                            random_noop=args.random_noop,
                            start_x=args.start_x,
                            frontier_prob=args.frontier_prob, offset=0)
    # 评估环境必须是独立的一套, 不能复用训练环境。
    # 评估故意不开随机起手: 完整游戏进 1-3 时就是关卡加载后立刻给第一个动作,
    # 等价于 noop=0, 这样选出来的 best_model 才正好是"能迁移"的那一份。
    eval_env = build_vec_env(n_envs, stage["env_id"], stage["single_stage"],
                             args.time_penalty, args.death_penalty,
                             random_noop=0, start_x=0, offset=100)
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
    if args.random_noop > 0:
        print(f"[随机起手] 每局开头随机 NOOP 0~{args.random_noop} 次")
    if args.start_x > 0:
        n_frontier = sum(1 for i in range(args.n_envs)
                         if i / args.n_envs < args.frontier_prob)
        print(f"[检查点] {n_frontier}/{args.n_envs} 个 worker 跑到 x>={args.start_x}"
              f" 时存档, 之后从那里开局")
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
                                args.time_penalty, args.death_penalty,
                                random_noop=0)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    model = PPO.load(str(model_path), env=env)
    print(f"[评估] {model_path}  on {stage['env_id']}")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total, max_x, flag = 0.0, 0, False
        lv_world, lv_stage = 0, 0          # 走到的最远关卡, 打完几关一眼可见
        start_x, end_x, steps = None, 0, 0
        milestone = 0                       # 上一次打过卡的百位刻度
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
            x = info[0]["x_pos"]                     # DummyVecEnv 的 info 是 list
            y, tleft = info[0]["y_pos"], info[0]["time"]
            if start_x is None:
                start_x = x
            if x > max_x:                            # 只在刷新最远距离时记录关卡
                max_x = x
                lv_world, lv_stage = info[0]["world"], info[0]["stage"]
            flag = flag or bool(info[0]["flag_get"])
            steps += 1
            end_x = x

            if args.show_x:
                # 里程碑: 每越过 100 打一条持久日志, 跑完能往上翻着对照画面。
                # y 是 255-屏幕像素y, 越大越高; y 恒定=站住了, 单调下降=在坠落。
                if x // 100 > milestone // 100:
                    milestone = x
                    print(f"\n[里程碑] x={x:5d}  y={y:4d}  时间={tleft:3d}")
                print(f"\r  x={x:5d}  y={y:4d}  时间={tleft:3d}  帧={steps:4d}",
                      end="", flush=True)

            if not args.headless:
                env.render()
                time.sleep(args.delay)
            done = bool(done[0])
        if args.show_x:
            print()                                  # 收掉 \r 那一行
        print(f"  第 {ep + 1} 局: 最远 x={max_x}  关卡={lv_world}-{lv_stage}  "
              f"reward={total:.1f}  {'* 举旗通关' if flag else '未通关'}")
        if args.show_x:
            print(f"    起点 x={start_x}  结束于 x={end_x}  ({steps} 帧)")


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
    # 每局开头随机 NOOP 0~N 次, 打散确定性轨迹。0 = 关闭(默认, 行为不变)。
    # 只在训练环境生效, 评估环境恒为 0。
    p.add_argument("--random-noop", type=int, default=0,
                   help="每局开头随机 NOOP 0~N 次, 破解确定性死锁; 0 = 关闭")
    # 检查点重启: 跑到 x>=N 且站稳时存档, 之后每局从那里开始。
    # 只在训练环境生效; 评估永远从关卡开头跑, 分数才代表真实全程表现。
    p.add_argument("--start-x", type=int, default=0,
                   help="检查点位置; 跑到 x>=N 后每局从那里开局; 0 = 关闭")
    p.add_argument("--frontier-prob", type=float, default=0.0,
                   help="多大比例的训练 worker 使用检查点(0~1)")
    p.add_argument("--eval", action="store_true", help="只评估, 不训练")
    # 边看画面边记位置: 用来给 --start-x 挑一个落在实地上、且有站立空间的点。
    p.add_argument("--show-x", action="store_true",
                   help="评估时实时打印马里奥的世界坐标 x/y (x 就是 --start-x 的刻度)")
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
