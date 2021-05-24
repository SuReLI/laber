import os


from stable_baselines3.dqn import DQN
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from stable_baselines3.sac_laber import SAC_LABER
from stable_baselines3.td3_laber import TD3_LABER
from stable_baselines3.sac_per import SAC_PER
from stable_baselines3.td3_per import TD3_PER
from stable_baselines3.sac_ger import SAC_GER
from stable_baselines3.td3_ger import TD3_GER

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
