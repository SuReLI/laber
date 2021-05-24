import pybullet_envs
from stable_baselines3 import TD3_PER

model = TD3_PER('MlpPolicy', 'MinitaurBulletEnv-v0', verbose=1, tensorboard_log="results/long_TD3_PER_MinitaurBullet/")
model.learn(total_timesteps=3000000)
