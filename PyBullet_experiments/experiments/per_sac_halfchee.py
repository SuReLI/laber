import pybullet_envs
from stable_baselines3 import SAC_PER

model = SAC_PER('MlpPolicy', 'HalfCheetahBulletEnv-v0', verbose=1, tensorboard_log="results/long_SAC_PER_HalfCheetahBullet/")
model.learn(total_timesteps=3000000)
