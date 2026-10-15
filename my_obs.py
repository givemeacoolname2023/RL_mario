import gym
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import  GrayScaleObservation,ResizeObservation
from my_wrapper import SkipFrameWrapper

def make_env(): #比较消耗CPU
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrameWrapper(env, skip=4)
    env = GrayScaleObservation(env,keep_dim=True)
    # env = ResizeObservation(env, (84, 84))
    return env
