import pybullet_envs
from stable_baselines3 import SAC_LABER

model = SAC_LABER('MlpPolicy', 'HumanoidBulletEnv-v0', verbose=1, tensorboard_log="results/long_SAC_LABER_HumanoidBullet/")
model.learn(total_timesteps=3000000)
