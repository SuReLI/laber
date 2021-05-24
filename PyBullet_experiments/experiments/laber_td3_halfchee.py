import pybullet_envs
from stable_baselines3 import TD3_LABER

model = TD3_LABER('MlpPolicy', 'HalfCheetahBulletEnv-v0', verbose=1, tensorboard_log="results/long_TD3_LABER_HalfCheetahBullet/")
model.learn(total_timesteps=3000000)
