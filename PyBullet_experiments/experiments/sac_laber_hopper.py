import pybullet_envs
from stable_baselines3 import SAC_LABER

model = SAC_LABER('MlpPolicy', 'HopperBulletEnv-v0', verbose=1, tensorboard_log="results/SAC_LABER_Hopper/")
model.learn(total_timesteps=3000000)
