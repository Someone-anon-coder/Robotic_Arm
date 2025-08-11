# eeg_robotic_arm_rl/envs/robotic_arm_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import time
from collections import deque

from config.simulation_config import SimConfig
from utils.simulation_utils import setup_simulation, get_joint_mappings
from utils.sensor_utils import get_link_indices_by_name, get_simulated_flex_sensors
from utils.reward_utils import calculate_reward

class RoboticArmEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode='human'):
        super().__init__()
        
        self.render_mode = render_mode
        self.config = SimConfig()
        self.client = None

        # --- Simulation Setup ---
        if self.render_mode == 'human':
            self.client, self.robot_id, self.glove_id = setup_simulation(self.config, p.GUI)
        else:
            self.client, self.robot_id, self.glove_id = setup_simulation(self.config, p.DIRECT)
            
        # --- Joint and Link Mappings ---
        self.robot_joint_map, self.robot_joint_map_inv = get_joint_mappings(self.robot_id)
        self.glove_joint_map, _ = get_joint_mappings(self.glove_id)
        self.controllable_joints = list(self.robot_joint_map.values())
        self.num_controllable_joints = len(self.controllable_joints)
        self.controllable_joint_names = list(self.robot_joint_map.keys())

        # Get link indices for sensors.
        # We need to explicitly list the links we're interested in for the flex sensors.
        finger_prefixes = ["index", "middle", "ring", "pinky", "thumb"]
        sensor_link_names = []
        for prefix in finger_prefixes:
            sensor_link_names.append(f"{prefix}_tracker_link")
            sensor_link_names.append(f"{prefix}_flexed_tracker_link")

        self.glove_sensor_link_map = get_link_indices_by_name(
            self.glove_id, sensor_link_names
        )
        self.glove_imu_link_idx = get_link_indices_by_name(self.glove_id, ["IMU_A_link_tracker"])["IMU_A_link_tracker"]
        self.robot_imu_link_idx = get_link_indices_by_name(self.robot_id, ["hand_base_link"])["hand_base_link"] # Using hand base as proxy
        
        # --- State Variables ---
        self.step_counter = 0
        self.prev_total_error = float('inf')
        self.imu_history = deque(maxlen=self.config.IMU_HISTORY_LENGTH)

        # --- Action and Observation Spaces ---
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_controllable_joints,), dtype=np.float32
        )

        # Observation Space:
        # 1. Flex Sensor Inputs (14)
        # 2. IMU History (10x3 = 30)
        # 3. Robot Joint Positions (27)
        # 4. Robot Joint Velocities (27)
        obs_space_size = 14 + (self.config.IMU_HISTORY_LENGTH * 3) + self.num_controllable_joints * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32
        )
        
        print(f"Environment Initialized. Observation Space Size: {obs_space_size}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_counter = 0
        self.prev_total_error = float('inf')
        
        # Reset glove to a random pose (for better training)
        target_angles = self.np_random.uniform(low=-0.5, high=1.5, size=p.getNumJoints(self.glove_id))
        for i, angle in enumerate(target_angles):
             p.resetJointState(self.glove_id, i, targetValue=angle)

        # Reset robot to initial state
        for i in self.controllable_joints:
            p.resetJointState(self.robot_id, i, targetValue=0, targetVelocity=0)
            
        # Initialize IMU history
        self.imu_history.clear()
        initial_imu_reading = np.zeros(3)
        for _ in range(self.config.IMU_HISTORY_LENGTH):
            self.imu_history.append(initial_imu_reading)
            
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.controllable_joints,
            controlMode=p.POSITION_CONTROL, # Using POSITION_CONTROL is often more stable
            targetPositions=self._action_to_target_positions(action)
        )
        
        p.stepSimulation()
        self.step_counter += 1
            
        observation = self._get_observation()
        reward, current_error = self._get_reward()
        terminated = self._check_termination(current_error)
        truncated = self.step_counter >= self.config.MAX_EPISODE_STEPS
        info = self._get_info(current_error=current_error)

        if terminated:
             reward += self.config.W_GOAL_ACHIEVED

        self.prev_total_error = current_error

        if self.render_mode == 'human':
            time.sleep(self.metadata['render_fps']**-1)
        
        return observation, reward, terminated, truncated, info

    def _action_to_target_positions(self, action):
        # Map normalized action [-1, 1] to joint limits
        # For now, let's use a simple mapping. A more precise one would use joint limits from URDF.
        # This is a placeholder for a more sophisticated mapping.
        return action * np.pi # Map to [-pi, pi] range

    def _get_observation(self):
        # 1. Simulated Flex Sensors
        flex_sensors = get_simulated_flex_sensors(self.glove_id, self.glove_sensor_link_map)
        
        # 2. IMU data
        glove_imu_state = p.getLinkState(self.glove_id, self.glove_imu_link_idx)
        robot_imu_state = p.getLinkState(self.robot_id, self.robot_imu_link_idx)
        glove_pos = np.array(glove_imu_state[0])
        robot_pos = np.array(robot_imu_state[0])
        imu_relative_pos = glove_pos - robot_pos
        self.imu_history.append(imu_relative_pos)
        
        # 3. Robot Joint States
        robot_states = p.getJointStates(self.robot_id, self.controllable_joints)
        robot_pos = [state[0] for state in robot_states]
        robot_vel = [state[1] for state in robot_states]
        
        return np.concatenate([
            flex_sensors / 1023.0, # Normalize
            np.array(self.imu_history).flatten(),
            robot_pos,
            robot_vel
        ]).astype(np.float32)

    def _get_reward(self):
        robot_states = p.getJointStates(self.robot_id, self.controllable_joints)
        glove_states = p.getJointStates(self.glove_id, self.controllable_joints)
        
        robot_pos = np.array([state[0] for state in robot_states])
        robot_vel = np.array([state[1] for state in robot_states])
        glove_pos = np.array([state[0] for state in glove_states])
        
        return calculate_reward(
            robot_pos,
            glove_pos,
            robot_vel,
            self.prev_total_error,
            self.controllable_joint_names,
            self.config
        )
    
    def _check_termination(self, current_error):
        return current_error < self.config.GOAL_ACHIEVED_THRESHOLD

    def _get_info(self, current_error=None):
        info = {}
        if current_error is not None:
            info['current_error'] = current_error
        return info

    def close(self):
        if self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
        self.client = None