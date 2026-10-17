from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack,VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from my_obs import make_env # 手写的构造环境包

def main():
    n_envs = 4

    # ---- 训练环境 ----
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)]) # 多进程
    vec_env = VecFrameStack(vec_env, n_stack=4,channels_order="last") # 帧堆叠
    vec_env = VecMonitor(vec_env)          # ★ 让 TensorBoard 有 rollout/ep_rew_mean
    
    # ---- 评估环境：必须是独立的一套，不能复用训练环境 ----
    eval_env = SubprocVecEnv([make_env(100 + i) for i in range(n_envs)])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(eval_env=eval_env,
                                best_model_save_path="./best_model",
                                log_path="./callback_logs",
                                eval_freq=100000//n_envs,    # 每 10 万 agent step 评估一次（=25000 次 env.step）
                                n_eval_episodes=n_envs,          # 每次评估 4 个 episode，episode（回合）= 从环境 reset 到结束（done）的完整一段过程。
                                deterministic=True,          # 评估时使用确定性策略, 直接选概率最大的action
                            )
    
 
    # 定期自动保存训练过程中的模型快照（每约 50 万 agent step 存一次），防止训练崩溃/中断导致成果丢失，并支持中断后续训或回溯。
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1), save_path="./checkpoints", name_prefix="mario"
    )

    """
    假如n_steps=2048, n_envs=4, 相当于在mario里, 每2048帧就是一个env。
    总样本帧数一共是2048*4, 而一个epoch里训练的完整数据就是2048*4帧
    batch_size要小于2048*4, 一个epoch里梯度更新的次数就是(2048*4)/batch_size
    
    """
    model = PPO("CnnPolicy", vec_env, verbose=1,tensorboard_log="logs",
                learning_rate=3e-4,  # 学习率
                n_steps=1024,   # 训练之前的收集阶段，每个环境收集多少步经验;和内存相关
                batch_size=256,  # 一个epoch里每次梯度更新需要的样本数；和显存相关:resize84×84 之后显存完全不是瓶颈了，可以放心加大
                n_epochs=4,    # 每个epoch训练所有的数据，一共4轮
                ent_coef=0.01,   # 每一点熵，就奖励0.01。数值越大，策略更随机，更爱探索。
                # target_kl=0.1,  # KL散度阈值，超过这个值就停止PPO训练,重新进入收集阶段
                gamma=0.99,   # 折扣因子，越接近1，越看重未来奖励
                )
    print(f"当前使用的设备是: {model.device}") 

    """
    total_timesteps 是总环境步数预算。每次 PPO 更新前，尽量收集 n_steps * n_envs 步。
    累计达到 total_timesteps 后停止。最后一次可能不满一批，而且可能不进行 PPO 训练。
    """
    model.learn(total_timesteps=5e6,callback=[eval_callback, checkpoint_callback])  # 训练500万步
    model.save("ppo_mario")

if __name__ == "__main__":
    main()