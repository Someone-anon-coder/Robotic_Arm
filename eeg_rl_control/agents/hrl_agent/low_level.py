import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

class ControllerAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # The controller's action space is the 27D motor command vector
        controller_action_space = env.action_space

        # The controller's observation space is the full 115D space (108 base + 7D subgoal)
        controller_observation_space = env.observation_space

        self.model = SAC(
            "MlpPolicy",
            controller_observation_space,
            action_space=controller_action_space,
            **self.config
        )
