import os
import numpy as np
import torch
import pybullet as p
from stable_baselines3.common.logger import configure

from environment.arm_env import ArmEnv
from agents.hrl_agent.high_level import ManagerAgent
from agents.hrl_agent.low_level import ControllerAgent
from config import HRL_CONFIG, MANAGER_CONFIG, CONTROLLER_CONFIG

def main():
    # --- Initialization ---
    log_path = os.path.join("./logs/hrl", "run")
    # The logger can be configured to output to console, tensorboard, etc.
    # "stdout" for console, "tensorboard" for tensorboard logging.
    sb3_logger = configure(log_path, ["stdout", "tensorboard"])

    env = ArmEnv(include_goal_in_obs=True, render_mode='none')
    manager = ManagerAgent(env, MANAGER_CONFIG)
    controller = ControllerAgent(env, CONTROLLER_CONFIG)

    manager.model.set_logger(sb3_logger)
    controller.model.set_logger(sb3_logger)

    global_step = 0
    episode_num = 0

    # --- Main Training Loop ---
    while global_step < HRL_CONFIG['total_timesteps']:
        episode_num += 1
        obs, info = env.reset()
        manager_state = obs[:108]

        episode_extrinsic_reward = 0
        episode_intrinsic_reward = 0
        episode_steps = 0

        manager_reward = 0

        done = False

        subgoal = manager.model.predict(manager_state, deterministic=False)[0]
        manager_state_start_of_macro = manager_state

        while not done:
            # --- Controller's Turn ---
            controller_state = np.concatenate([manager_state, subgoal])
            controller_action, _ = controller.model.predict(controller_state, deterministic=False)

            next_obs, extrinsic_reward, terminated, truncated, info = env.step(controller_action)
            done = terminated or truncated

            # --- Reward Calculation ---
            episode_extrinsic_reward += extrinsic_reward
            manager_reward += extrinsic_reward

            # Intrinsic reward for the controller
            wrist_link_state = p.getLinkState(env.agent_arm, env.agent_link_map['hand_base_link'])
            current_wrist_pose = np.concatenate([wrist_link_state[0], wrist_link_state[1]])
            intrinsic_reward = -np.linalg.norm(current_wrist_pose - subgoal)
            episode_intrinsic_reward += intrinsic_reward

            # --- Experience Storing ---
            next_manager_state = next_obs[:108]
            next_controller_state = np.concatenate([next_manager_state, subgoal])
            controller.model.replay_buffer.add(
                controller_state, next_controller_state, controller_action, intrinsic_reward, done, [info]
            )

            # --- Manager's Turn ---
            if (global_step + 1) % HRL_CONFIG['manager_freq'] == 0 or done:
                manager.model.replay_buffer.add(
                    manager_state_start_of_macro, next_manager_state, subgoal, manager_reward, done, [info]
                )
                subgoal = manager.model.predict(next_manager_state, deterministic=False)[0]
                manager_reward = 0
                manager_state_start_of_macro = next_manager_state

            manager_state = next_manager_state

            # --- Training ---
            controller.model.train(gradient_steps=1)
            if (global_step + 1) % HRL_CONFIG['manager_freq'] == 0:
                manager.model.train(gradient_steps=1)

            # --- Logging & Updates ---
            if (global_step + 1) % HRL_CONFIG['log_interval'] == 0 and episode_steps > 0:
                avg_ext_reward = episode_extrinsic_reward / episode_steps
                avg_int_reward = episode_intrinsic_reward / episode_steps
                # The SB3 logger will handle printing to stdout and writing to TensorBoard
                manager.model.logger.record('hrl/extrinsic_reward', avg_ext_reward)
                manager.model.logger.record('hrl/intrinsic_reward', avg_int_reward)
                manager.model.logger.record('hrl/episode', episode_num)
                manager.model.logger.dump(step=global_step)
                episode_extrinsic_reward = 0
                episode_intrinsic_reward = 0
                episode_steps = 0

            global_step += 1
            episode_steps += 1
            obs = next_obs

if __name__ == '__main__':
    main()
