import gymnasium as gym
from environment.arm_env import ArmEnv
from agents.hrl_agent.high_level import ManagerAgent
from agents.hrl_agent.low_level import ControllerAgent

def main():
    # Instantiate the environment with the HRL flag
    env = ArmEnv(include_goal_in_obs=True, render_mode=None)

    # Define separate placeholder config dictionaries
    manager_config = {
        "policy": "MlpPolicy",
        "verbose": 0,
    }
    controller_config = {
        "policy": "MlpPolicy",
        "verbose": 0,
    }

    # Instantiate the Manager
    manager = ManagerAgent(env, manager_config)

    # Instantiate the Controller
    controller = ControllerAgent(env, controller_config)

    # Print Verification Info
    print(f"Manager Observation Space: {manager.model.observation_space.shape}")
    print(f"Manager Action Space: {manager.model.action_space.shape}")
    print(f"Controller Observation Space: {controller.model.observation_space.shape}")
    print(f"Controller Action Space: {controller.model.action_space.shape}")

    env.close()

if __name__ == "__main__":
    main()
