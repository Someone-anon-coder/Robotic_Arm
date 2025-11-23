import torch
import numpy as np
import yaml
import pybullet as p
from eeg_rl_control.simulation.environment import BiomimeticArmEnv
from eeg_rl_control.models.agent_wrappers import WristShoulderAgent, FingerAgent, ThumbPalmAgent
from eeg_rl_control.models.sac_core import ReplayBuffer
from eeg_rl_control.simulation.reward_function import RewardCalculator

def randomize_ghost_pose(env, joint_limits):
    """Sets the ghost arm to a random valid pose."""
    joint_targets = {}
    for joint_group, limits in joint_limits.items():
        if joint_group.endswith('_joints'):
            action_map = env.config['action_maps'][joint_group]
            for joint_name in action_map:
                joint_targets[joint_name] = np.random.uniform(low=limits['min'], high=limits['max'])
    for joint_name, target_angle in joint_targets.items():
        if joint_name in env.joint_map_ghost:
            env.p.resetJointState(env.ghost_id, env.joint_map_ghost[joint_name], target_angle)

def main():
    # 1. Initialization
    with open('eeg_rl_control/config/rl_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = BiomimeticArmEnv('eeg_rl_control/config/sim_params.yaml')
    reward_calculator = RewardCalculator()

    wrist_shoulder_agent = WristShoulderAgent('eeg_rl_control/config/rl_config.yaml')
    finger_agent = FingerAgent('eeg_rl_control/config/rl_config.yaml')
    thumb_palm_agent = ThumbPalmAgent('eeg_rl_control/config/rl_config.yaml')

    agents = {
        'wrist_shoulder': wrist_shoulder_agent,
        'fingers': finger_agent,
        'thumb_palm': thumb_palm_agent
    }

    replay_buffers = {
        'wrist_shoulder': ReplayBuffer(100000),
        'fingers': ReplayBuffer(100000),
        'thumb_palm': ReplayBuffer(100000)
    }

    # 2. Training Loop
    for i_episode in range(1, 1001):
        obs = env.reset()
        randomize_ghost_pose(env, config['joint_limits'])
        episode_reward = 0
        q_values = {agent_name: [] for agent_name in agents.keys()}

        states = {}
        for agent_name, agent in agents.items():
            if agent_name == 'wrist_shoulder':
                state = agent.obs_to_tensor(obs['agent_wrist_shoulder']).numpy()
            elif agent_name == 'fingers':
                state = agent.obs_to_tensor(obs['agent_fingers_imrp']).numpy()
            else: # thumb_palm
                state = agent.obs_to_tensor(obs['agent_thumb_palm']).numpy()
            states[agent_name] = state

        for t in range(200): # Max steps per episode
            # Get Actions
            actions_np = {}
            actions = {}
            for agent_name, agent in agents.items():
                state = states[agent_name]
                action = agent.sac_agent.select_action(state)
                actions_np[agent_name] = action
                actions.update(agent.action_to_dict(action))

            # Step Environment
            next_obs = env.step(actions)

            # Calculate Reward
            rewards = reward_calculator.compute_reward(next_obs)
            episode_reward += sum(rewards.values())

            # Store Transitions
            done = (t == 199)
            next_states = {}
            for agent_name, agent in agents.items():
                if agent_name == 'wrist_shoulder':
                    next_state = agent.obs_to_tensor(next_obs['agent_wrist_shoulder']).numpy()
                elif agent_name == 'fingers':
                    next_state = agent.obs_to_tensor(next_obs['agent_fingers_imrp']).numpy()
                else: # thumb_palm
                    next_state = agent.obs_to_tensor(next_obs['agent_thumb_palm']).numpy()
                next_states[agent_name] = next_state
                replay_buffers[agent_name].push(states[agent_name], actions_np[agent_name], rewards[agent_name], next_state, done)

            obs = next_obs
            states = next_states

            # Train Agents
            for agent_name, agent in agents.items():
                q_val = agent.sac_agent.get_q_values(states[agent_name], actions_np[agent_name])
                q_values[agent_name].append(q_val)
                if len(replay_buffers[agent_name]) > config['sac_hyperparameters']['batch_size']:
                    agent.sac_agent.train(replay_buffers[agent_name], config['sac_hyperparameters']['batch_size'])

        # Logging
        if i_episode % 10 == 0:
            avg_q_str = ", ".join([f"Avg Q-{name}: {np.mean(vals):.2f}" for name, vals in q_values.items()])
            print(f"Episode: {i_episode}, Total Reward: {episode_reward}, {avg_q_str}")

        # Saving
        if i_episode % 100 == 0:
            for agent_name, agent in agents.items():
                torch.save(agent.sac_agent.policy.state_dict(), f'{agent_name}_policy_episode_{i_episode}.pth')
                torch.save(agent.sac_agent.critic.state_dict(), f'{agent_name}_critic_episode_{i_episode}.pth')

if __name__ == '__main__':
    main()