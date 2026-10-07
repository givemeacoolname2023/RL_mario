import gym
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from my_obs import make_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

env = make_env()
env = DummyVecEnv([lambda: env])
env =  VecFrameStack(env, n_stack=4,channels_order="last")
model = PPO.load("best_model/best_model.zip",env=env)

obs = env.reset()
for i in range(1000):
    obs = obs.copy()  # Copy the observation to avoid modifying the original
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()