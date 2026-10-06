import gym
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import  GrayScaleObservation,ResizeObservation
from my_wrapper import SkipFrameWrapper
from my_obs import make_env

def main():
    env = make_env()
    model = PPO("CnnPolicy", env, verbose=1,tensorboard_log="logs")
    print(f"当前使用的设备是: {model.device}") 
    model.learn(total_timesteps=10000)
    model.save("ppo_mario")

if __name__ == "__main__":
    main()