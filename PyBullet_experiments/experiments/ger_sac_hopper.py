import pybullet_envs
from stable_baselines3 import SAC_GER

model = SAC_GER('MlpPolicy', 'HopperBulletEnv-v0', verbose=1, tensorboard_log="results/long_SAC_GER_HopperBullet/")
model.learn(total_timesteps=3000000)
