import pybullet_envs
from stable_baselines3 import SAC_GER

model = SAC_GER('MlpPolicy', 'HalfCheetahBulletEnv-v0', verbose=1, tensorboard_log="results/long_SAC_GER_HalfCheetahBullet/")
model.learn(total_timesteps=3000000)
