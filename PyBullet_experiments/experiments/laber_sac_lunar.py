import pybullet_envs
from stable_baselines3 import SAC_LABER

model = SAC_LABER('MlpPolicy', 'LunarLanderContinuous-v2', verbose=1, tensorboard_log="results/medium_SAC_LABER_LunarLanderContinuous/")
model.learn(total_timesteps=1200000)
