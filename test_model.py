import time
from stable_baselines3 import PPO
from my_obs import make_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

env = DummyVecEnv([make_env()])
env =  VecFrameStack(env, n_stack=4,channels_order="last")
model = PPO.load("models\\1-3\\best_model.zip",env=env)

obs = env.reset()
for i in range(2000):
    obs = obs.copy()  # Copy the observation to avoid modifying the original
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    # time.sleep(0.05)      # ← 新增这一行
    if done:
      # obs = env.reset()
      break