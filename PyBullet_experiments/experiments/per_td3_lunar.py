import pybullet_envs
from stable_baselines3 import TD3_PER

model = TD3_PER('MlpPolicy', 'LunarLanderContinuous-v2', verbose=1, tensorboard_log="results/medium_TD3_PER_LunarLanderContinuous/")
model.learn(total_timesteps=1200000)
