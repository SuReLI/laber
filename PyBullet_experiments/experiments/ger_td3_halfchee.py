import autograd_hacks
import pybullet_envs
from stable_baselines3 import TD3_GER

model = TD3_GER('MlpPolicy', 'HalfCheetahBulletEnv-v0', verbose=1, tensorboard_log="results/long_TD3_GER_HalfCheetahBullet/")
model.learn(total_timesteps=3000000)
