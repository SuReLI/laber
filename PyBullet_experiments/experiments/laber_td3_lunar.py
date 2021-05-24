import pybullet_envs
from stable_baselines3 import TD3_LABER

model = TD3_LABER('MlpPolicy', 'LunarLanderContinuous-v2', verbose=1, tensorboard_log="results/medium_TD3_LABER_LunarLanderContinuous/")
model.learn(total_timesteps=1200000)
