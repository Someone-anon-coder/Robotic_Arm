import os
import time
import numpy as np
import pybullet as p

from eeg_rl_control.environment.arm_env import ArmEnv
from eeg_rl_control.agents.hrl_agent.high_level import ManagerAgent
from eeg_rl_control.agents.hrl_agent.low_level import ControllerAgent
from eeg_rl_control.config import HRL_CONFIG, MANAGER_CONFIG, CONTROLLER_CONFIG

def unnormalize_subgoal(normalized_subgoal_pos, bounding_box):
    """Un-normalizes a subgoal from [-1, 1] to world coordinates."""
    center = (bounding_box['max'] + bounding_box['min']) / 2
    half_range = (bounding_box['max'] - bounding_box['min']) / 2
    return center + normalized_subgoal_pos * half_range

def main():
    # Define the bounding box for the subgoal's target position
    WRIST_POSITION_BOUNDING_BOX = {
        'min': np.array([0.0, -0.4, 0.2]),
        'max': np.array([0.8, 0.4, 1.0])
    }

    # Create directories for logging and model saving
    os.makedirs(HRL_CONFIG["log_path"], exist_ok=True)
    os.makedirs(HRL_CONFIG["model_save_path"], exist_ok=True)
    os.makedirs(MANAGER_CONFIG["tensorboard_log"], exist_ok=True)
    os.makedirs(CONTROLLER_CONFIG["tensorboard_log"], exist_ok=True)

    # Instantiate the environment
    env = ArmEnv(render_mode='none', include_goal_in_obs=True)

    # Instantiate the agents
    manager = ManagerAgent(env, MANAGER_CONFIG)
    controller = ControllerAgent(env, CONTROLLER_CONFIG)

    # Initialize training variables
    obs, _ = env.reset()
    # Initialize subgoal to a valid shape; the manager will overwrite this shortly
    subgoal = np.zeros(manager.model.action_space.shape)
    episode_manager_reward = 0
    episode_start_step = 0
    manager_obs_start_of_macro = None

    print("Initialization complete. Starting training loop...")
    start_time = time.time()

    for step in range(HRL_CONFIG["total_training_steps"]):
        # --- Manager's Decision Step ---
        if step % HRL_CONFIG["manager_action_frequency"] == 0:
            # The manager observes the base state (first 108 elements)
            manager_obs = obs[:108]
            subgoal, _ = manager.model.predict(manager_obs, deterministic=False)
            # Store the state where the manager made its decision
            manager_obs_start_of_macro = manager_obs

        # --- Controller's Decision Step ---
        # The controller observes the base state AND the current subgoal
        controller_obs = np.concatenate([obs[:108], subgoal])
        action, _ = controller.model.predict(controller_obs, deterministic=False)

        # --- Environment Interaction ---
        next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_manager_reward += extrinsic_reward

        # --- Intrinsic Reward Calculation (for Controller) ---
        # Get current wrist position
        wrist_link_state = p.getLinkState(env.agent_arm, env.agent_link_map['hand_base_link'])
        current_wrist_pos = np.array(wrist_link_state[0])

        # Un-normalize the manager's subgoal to get a target world position
        # The subgoal from the manager is 7D: 3D position + 4D orientation
        subgoal_pos_normalized = subgoal[:3]
        target_wrist_pos = unnormalize_subgoal(subgoal_pos_normalized, WRIST_POSITION_BOUNDING_BOX)

        # Calculate distance and intrinsic reward
        distance = np.linalg.norm(current_wrist_pos - target_wrist_pos)
        intrinsic_reward = -distance

        # --- Store Controller Experience ---
        next_controller_obs = np.concatenate([next_obs[:108], subgoal])
        controller.model.replay_buffer.add(
            controller_obs, next_controller_obs, action, intrinsic_reward, done, [info]
        )

        # --- Store Manager Experience ---
        if (step + 1) % HRL_CONFIG["manager_action_frequency"] == 0 or done:
            next_manager_obs = next_obs[:108]
            # The manager's reward is the sum of extrinsic rewards during its macro-action
            manager.model.replay_buffer.add(
                manager_obs_start_of_macro, next_manager_obs, subgoal, episode_manager_reward, done, [info]
            )
            # Reset the accumulator for the next macro-action
            episode_manager_reward = 0

        # --- Update Policies ---
        # Note: stable-baselines3's SAC `train` method handles buffer sampling and learning rate schedules.
        # By default, it trains one gradient step per call.
        controller.model.train(gradient_steps=1, batch_size=CONTROLLER_CONFIG["batch_size"])

        if (step + 1) % HRL_CONFIG["manager_action_frequency"] == 0:
            manager.model.train(gradient_steps=HRL_CONFIG["manager_action_frequency"], batch_size=MANAGER_CONFIG["batch_size"])

        # Update obs
        obs = next_obs

        # --- Handle Episode End ---
        if done:
            print(f"Step: {step}/{HRL_CONFIG['total_training_steps']}, Episode finished after {step - episode_start_step + 1} steps.")
            obs, _ = env.reset()
            episode_manager_reward = 0
            episode_start_step = step + 1

        # --- Periodic Model Saving ---
        if (step + 1) % 50000 == 0:
            print(f"--- Saving model at step {step + 1} ---")
            manager.model.save(os.path.join(HRL_CONFIG["model_save_path"], f"manager_model_step_{step+1}.zip"))
            controller.model.save(os.path.join(HRL_CONFIG["model_save_path"], f"controller_model_step_{step+1}.zip"))


if __name__ == "__main__":
    main()
