import pybullet_envs
from stable_baselines3 import SAC_GER

model = SAC_GER('MlpPolicy', 'LunarLanderContinuous-v2', verbose=1, tensorboard_log="results/medium_SAC_GER_LunarLanderContinuous/")
model.learn(total_timesteps=1200000)
