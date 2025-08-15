import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

class ManagerAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # The manager's action space is the 7D subgoal for the wrist
        manager_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # The manager's observation space is the base environment's observation space
        manager_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(108,), dtype=np.float32)

        self.model = SAC(
            "MlpPolicy",
            manager_observation_space,
            action_space=manager_action_space,
            **self.config
        )
