# eeg_robotic_arm_rl/envs/robotic_arm_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import time

from config.simulation_config import SimConfig
from utils.simulation_utils import setup_simulation, get_joint_mappings

class RoboticArmEnv(gym.Env):
    """Custom Gymnasium environment for the EEG Robotic Arm."""
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode='human'):
        super().__init__()
        
        self.render_mode = render_mode
        self.client = None

        # Set up the simulation
        if self.render_mode == 'human':
            self.client, self.robot_id, self.glove_id = setup_simulation(SimConfig, p.GUI)
        else:
            self.client, self.robot_id, self.glove_id = setup_simulation(SimConfig, p.DIRECT)
            
        # Get joint mappings for controllable joints
        self.robot_joint_map, _ = get_joint_mappings(self.robot_id)
        self.glove_joint_map, _ = get_joint_mappings(self.glove_id)

        # Ensure the joint names match between robot and glove for control
        assert self.robot_joint_map.keys() == self.glove_joint_map.keys(), \
            "Robot and Glove must have the same controllable joint names."
            
        self.controllable_joints = list(self.robot_joint_map.values())
        self.num_controllable_joints = len(self.controllable_joints)

        # Define action space: normalized torque for each controllable joint
        # Values are typically in [-1, 1] and will be scaled
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_controllable_joints,), dtype=np.float32
        )

        # Define observation space
        # We'll observe:
        # 1. Robot's current joint positions (27 values)
        # 2. Robot's current joint velocities (27 values)
        # 3. Glove's target joint positions (27 values)
        obs_space_size = self.num_controllable_joints * 3 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32
        )
        
        print(f"Environment Initialized. Controllable Joints: {self.num_controllable_joints}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- 1. Set a new target pose for the glove ---
        # For now, let's make a simple "fist" like pose as a target
        target_angles = np.zeros(p.getNumJoints(self.glove_id))
        for joint_name, joint_index in self.glove_joint_map.items():
            if 'mcp_flex' in joint_name or 'pip' in joint_name or 'dip' in joint_name:
                target_angles[joint_index] = 1.0 # Flex the main finger joints
            if 'thumb_mcp_flex' in joint_name or 'thumb_ip' in joint_name:
                target_angles[joint_index] = 0.8 # Flex the thumb

        for i, angle in enumerate(target_angles):
             p.resetJointState(self.glove_id, i, targetValue=angle)

        # --- 2. Reset the robotic arm to its initial state ---
        for i in self.controllable_joints:
            p.resetJointState(self.robot_id, i, targetValue=0, targetVelocity=0)
            
        # --- 3. Get initial observation ---
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        # --- 1. Apply action to the robot ---
        # The action is a normalized value in [-1, 1]. We need to scale it.
        # Let's assume a max torque for now.
        max_torque = 10.0 # This can be tuned later
        scaled_action = action * max_torque
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.controllable_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=scaled_action
        )
        
        # --- 2. Step the simulation ---
        p.stepSimulation()
        if self.render_mode == 'human':
            time.sleep(self.metadata['render_fps']**-1)
            
        # --- 3. Get results ---
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = False # We will define termination conditions later
        truncated = False # e.g., max steps per episode
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        robot_states = p.getJointStates(self.robot_id, self.controllable_joints)
        robot_pos = [state[0] for state in robot_states]
        robot_vel = [state[1] for state in robot_states]
        
        glove_states = p.getJointStates(self.glove_id, self.controllable_joints)
        glove_pos = [state[0] for state in glove_states]
        
        return np.concatenate([robot_pos, robot_vel, glove_pos]).astype(np.float32)

    def _get_reward(self):
        # Simple reward: negative distance to the target pose
        obs = self._get_observation()
        robot_pos = obs[:self.num_controllable_joints]
        glove_pos = obs[self.num_controllable_joints*2:]
        
        # L2 distance (Euclidean distance)
        position_error = np.linalg.norm(robot_pos - glove_pos)
        
        return -position_error

    def _get_info(self):
        # We can add debug info here later
        return {}

    def close(self):
        if self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
        self.client = None