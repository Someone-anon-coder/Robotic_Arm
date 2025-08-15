import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from environment.arm_env import ArmEnv

# Create a custom environment wrapper for the ControllerAgent
class ControllerEnvWrapper(gym.Env):
    def __init__(self, base_env: ArmEnv):
        super().__init__()
        self.base_env = base_env

        # The controller's observation space is the full 115D space (108 base + 7D subgoal)
        self.observation_space = base_env.observation_space

        # The controller's action space is the 27D motor command vector
        self.action_space = base_env.action_space

    def reset(self, seed=None, options=None):
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action):
        return self.base_env.step(action)

class ControllerAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Pass the custom wrapper environment to SAC
        controller_wrapped_env = ControllerEnvWrapper(env)
        self.model = SAC(
            env=controller_wrapped_env,
            **self.config
        )