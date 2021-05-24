import pybullet_envs
from stable_baselines3 import SAC_PER

model = SAC_PER('MlpPolicy', 'LunarLanderContinuous-v2', verbose=1, tensorboard_log="results/medium_SAC_PER_LunarLanderContinuous/")
model.learn(total_timesteps=1200000)
