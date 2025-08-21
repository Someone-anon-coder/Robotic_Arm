import os
import numpy as np
import pybullet as p
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from eeg_rl_control.environment.arm_env import ArmEnv
from eeg_rl_control.agents.hrl_agent.high_level import ManagerAgent
from eeg_rl_control.agents.hrl_agent.low_level import ControllerAgent
from eeg_rl_control.config_hrl import MANAGER_CONFIG, CONTROLLER_CONFIG, HRL_TRAIN_CONFIG

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.global_step = 0
        self.episode_num = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.global_step += 1
        return True

    def on_episode_end(self, reward, logger, global_step):
        logger.record("episode/reward", reward)
        logger.record("episode/episode_num", self.episode_num)
        self.episode_num += 1
        logger.dump(step=global_step)

def get_expert_action(env):
    """
    A simple PID-like expert that tries to match the ghost's joint positions.
    """
    ghost_joint_states = p.getJointStates(env.ghost_arm, range(p.getNumJoints(env.ghost_arm)))
    # We only care about the controllable joints of the agent
    ghost_joint_positions = [state[0] for state in ghost_joint_states if p.getJointInfo(env.ghost_arm, state[0])[2] != p.JOINT_FIXED]


    agent_joint_states = p.getJointStates(env.agent_arm, env.agent_controllable_joints)
    agent_joint_positions = [state[0] for state in agent_joint_states]

    # This is a simplification. A real expert would need a proper mapping
    # between ghost joints and agent joints if they are not 1-to-1.
    # Assuming the first N joints of the ghost correspond to the agent's controllable joints.
    num_agent_joints = len(agent_joint_positions)
    motor_action = [g - a for g, a in zip(ghost_joint_positions[:num_agent_joints], agent_joint_positions)]

    return np.clip(motor_action, -1.0, 1.0)


def main():
    # --- Setup ---
    os.makedirs(HRL_TRAIN_CONFIG["model_save_path"], exist_ok=True)
    os.makedirs(HRL_TRAIN_CONFIG["tensorboard_log"], exist_ok=True)

    env = ArmEnv(render_mode='none', include_goal_in_obs=True)

    # --- Agents ---
    # The agents' internal SB3 models need a VecEnv, so we wrap the base env.
    vec_env = DummyVecEnv([lambda: env])
    manager = ManagerAgent(env=vec_env, config=MANAGER_CONFIG)
    controller = ControllerAgent(env=vec_env, config=CONTROLLER_CONFIG)

    # --- Logger ---
    new_logger = configure(HRL_TRAIN_CONFIG["tensorboard_log"], ["stdout", "tensorboard"])
    manager.model.set_logger(new_logger)
    controller.model.set_logger(new_logger)
    callback = TensorboardCallback()

    # --- Training Loop ---
    global_step = 0
    episode_num = 0

    while global_step < HRL_TRAIN_CONFIG["total_timesteps"]:
        episode_num += 1
        obs, info = env.reset()
        base_obs = obs

        manager_obs_start = base_obs
        subgoal = manager.model.predict(manager_obs_start, deterministic=True)[0]

        done = False
        episode_reward = 0
        accumulated_extrinsic_reward = 0

        while not done:
            # --- Manager's Turn ---
            if env.step_counter > 0 and env.step_counter % HRL_TRAIN_CONFIG["manager_update_freq_steps"] == 0:
                manager.model.replay_buffer.add(
                    manager_obs_start, base_obs, subgoal, accumulated_extrinsic_reward, float(done), [info]
                )

                subgoal = manager.model.predict(base_obs, deterministic=True)[0]
                manager_obs_start = base_obs
                accumulated_extrinsic_reward = 0

            # --- Controller's Turn ---
            controller_obs = np.concatenate([base_obs, subgoal])

            in_demo_range = any(start <= global_step < end for start, end in HRL_TRAIN_CONFIG["demo_steps_ranges"])
            if in_demo_range:
                motor_action = get_expert_action(env)
            else:
                motor_action, _ = controller.model.predict(controller_obs, deterministic=False) # Use stochastic for exploration

            next_obs, extrinsic_reward, terminated, truncated, info = env.step(motor_action)
            done = terminated or truncated
            next_base_obs = next_obs

            agent_wrist_pos = p.getLinkState(env.agent_arm, env.agent_link_map['hand_base_link'])[0]
            # Assuming the subgoal is the desired wrist pose [x,y,z,qx,qy,qz,qw]
            subgoal_pos = subgoal[:3]
            intrinsic_reward = -np.linalg.norm(np.array(agent_wrist_pos) - np.array(subgoal_pos))

            next_controller_obs = np.concatenate([next_base_obs, subgoal])
            controller.model.replay_buffer.add(
                controller_obs, next_controller_obs, motor_action, intrinsic_reward, float(done), [info]
            )

            accumulated_extrinsic_reward += extrinsic_reward

            if global_step > CONTROLLER_CONFIG["learning_starts"]:
                controller.model.train(gradient_steps=1)

            if global_step > MANAGER_CONFIG["learning_starts"] and (env.step_counter % HRL_TRAIN_CONFIG["manager_update_freq_steps"] == 0):
                 manager.model.train(gradient_steps=HRL_TRAIN_CONFIG["manager_update_freq_steps"])


            base_obs = next_base_obs
            episode_reward += extrinsic_reward
            global_step += 1

            if global_step % HRL_TRAIN_CONFIG["model_save_freq_steps"] == 0:
                manager_path = os.path.join(HRL_TRAIN_CONFIG["model_save_path"], f"manager_{global_step}_steps.zip")
                controller_path = os.path.join(HRL_TRAIN_CONFIG["model_save_path"], f"controller_{global_step}_steps.zip")
                manager.model.save(manager_path)
                controller.model.save(controller_path)
                new_logger.info(f"Saved models at step {global_step}")

        # Log episode stats
        callback.on_episode_end(episode_reward, new_logger, global_step)
        new_logger.record("hrl/manager_accumulated_reward", accumulated_extrinsic_reward)
        new_logger.record("hrl/last_intrinsic_reward", intrinsic_reward)
        new_logger.dump(step=global_step)


    env.close()

if __name__ == '__main__':
    main()
