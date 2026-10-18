import gym
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import  GrayScaleObservation,ResizeObservation
from my_wrapper import SkipFrameWrapper, MarioRewardWrapper

def make_env(rank=0): #比较消耗CPU。现在是闭包了
    """返回 thunk, 供 SubprocVecEnv / DummyVecEnv 调用。"""
    # 多进程需要「造环境的函数」，不是「环境」
    def _init():    # gym.make() 不能直接在多进程中调用，否则会报错,改为返回函数
        env = gym_super_mario_bros.make('SuperMarioBros-v2')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrameWrapper(env, skip=4)
        env = MarioRewardWrapper(env,time_penalty=0.01)   # 新增奖励重构
        env = GrayScaleObservation(env,keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        return env
    
    return _init
