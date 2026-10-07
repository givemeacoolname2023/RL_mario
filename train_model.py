import gym
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import  GrayScaleObservation,ResizeObservation
from my_wrapper import SkipFrameWrapper
from my_obs import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

def main():
    # env = make_env()
    vec_env = SubprocVecEnv([lambda: make_env() for _ in range(4)]) # 多进程
    vec_env = VecFrameStack(vec_env, n_stack=4,channels_order="last") # 帧堆叠

    eval_callback = EvalCallback(eval_env=vec_env, best_model_save_path="./best_model",
                                 log_path="./callback_logs", eval_freq=10000//4)

    """
    假如n_steps=2048, n_envs=4, 相当于在mario里, 每2048帧就是一个env。
    总样本帧数一共是2048*4, 而一个epoch里训练的完整数据就是2048*4帧
    batch_size要小于2048*4, 一个epoch里梯度更新的次数就是(2048*4)/batch_size
    
    """
    model = PPO("CnnPolicy", vec_env, verbose=1,tensorboard_log="logs",
                learning_rate=1e-4,  # 学习率
                n_steps=2048,   # 训练之前的收集阶段，每个环境收集多少步经验
                batch_size=256,  # 一个epoch里每次梯度更新需要的样本数
                n_epochs=4,    # 每个epoch训练所有的数据，一共10轮
                ent_coef=0.1,   # 每一点熵，就奖励0.1。数值越大，策略更随机，更爱探索。
                # target_kl=0.1,  # KL散度阈值，超过这个值就停止PPO训练,重新进入收集阶段
                gamma=0.95,   # 折扣因子，越接近1，越看重未来奖励
                )
    print(f"当前使用的设备是: {model.device}") 

    """
    total_timesteps 是总环境步数预算。每次 PPO 更新前，尽量收集 n_steps * n_envs 步。
    累计达到 total_timesteps 后停止。最后一次可能不满一批，而且可能不进行 PPO 训练。
    """
    model.learn(total_timesteps=1e6,callback=eval_callback)  # 训练100万步
    model.save("ppo_mario")

if __name__ == "__main__":
    main()