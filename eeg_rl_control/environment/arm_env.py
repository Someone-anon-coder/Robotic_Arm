import gymnasium as gym
import numpy as np
import pybullet as p
import collections
import math
import pybullet_data
import os

class ArmEnv(gym.Env):
    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        datapath = pybullet_data.getDataPath()
        p.setAdditionalSearchPath('urdf')
        self.plane = p.loadURDF(os.path.join(datapath, "plane.urdf"))

        # Load arms
        self.agent_arm = p.loadURDF("robotic_arm.urdf", [0, 0, 0], useFixedBase=True)
        self.ghost_arm = p.loadURDF("glove.urdf", [1, 0, 0], useFixedBase=True)

        self._build_joint_maps()

        # Define Action and Observation Spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32)

        observation_dim = 108
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        self.historical_imu_data = collections.deque(maxlen=10)

        # Flex sensor link pairs
        self.flex_sensor_pairs = [
            # Finger flexion sensors (tip to base)
            ('index_tracker_link', 'index_flexed_tracker_link'),
            ('middle_tracker_link', 'middle_flexed_tracker_link'),
            ('ring_tracker_link', 'ring_flexed_tracker_link'),
            ('pinky_tracker_link', 'pinky_flexed_tracker_link'),
            ('thumb_tracker_link', 'thumb_flexed_tracker_link'),

            # Palm to finger base sensors
            ('hand_base_link', 'index_flexed_tracker_link'),
            ('hand_base_link', 'middle_flexed_tracker_link'),
            ('hand_base_link', 'ring_flexed_tracker_link'),
            ('hand_base_link', 'pinky_flexed_tracker_link'),
            ('hand_base_link', 'thumb_flexed_tracker_link'),

            # Fingertip to thumb tip sensors
            ('index_tracker_link', 'thumb_tracker_link'),
            ('middle_tracker_link', 'thumb_tracker_link'),
            ('ring_tracker_link', 'thumb_tracker_link'),
            ('pinky_tracker_link', 'thumb_tracker_link'),
        ]


    def _get_observation(self):
        # 1. Agent Proprioception
        joint_states = p.getJointStates(self.agent_arm, self.agent_controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        proprioception = np.concatenate([joint_positions, joint_velocities]).astype(np.float32)

        # 2. Ghost IMU Data
        imu_link_state = p.getLinkState(self.ghost_arm, self.ghost_link_map['IMU_A_link_tracker'], computeLinkVelocity=1)
        linear_velocity = np.array(imu_link_state[6]).astype(np.float32)
        angular_velocity = np.array(imu_link_state[7]).astype(np.float32)
        orientation = np.array(imu_link_state[1]).astype(np.float32)
        imu_data = np.concatenate([linear_velocity, angular_velocity, orientation])

        # 3. Historical Data
        self.historical_imu_data.append(linear_velocity)
        historical_data = np.array(self.historical_imu_data).flatten().astype(np.float32)

        # 4. Simulated Flex Sensors
        flex_distances = []
        for link1_name, link2_name in self.flex_sensor_pairs:
            link1_state = p.getLinkState(self.ghost_arm, self.ghost_link_map[link1_name])
            link2_state = p.getLinkState(self.ghost_arm, self.ghost_link_map[link2_name])
            pos1 = np.array(link1_state[0])
            pos2 = np.array(link2_state[0])
            distance = np.linalg.norm(pos1 - pos2)
            flex_distances.append(distance)
        flex_distances = np.array(flex_distances).astype(np.float32)

        return np.concatenate([proprioception, imu_data, historical_data, flex_distances])

    def _build_joint_maps(self):
        self.agent_joint_map = {p.getJointInfo(self.agent_arm, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.agent_arm))}
        self.ghost_link_map = {p.getJointInfo(self.ghost_arm, i)[12].decode('utf-8'): i for i in range(p.getNumJoints(self.ghost_arm))}
        self.agent_controllable_joints = [v for k,v in self.agent_joint_map.items() if k.endswith('_joint')]

    def _set_random_ghost_pose(self):
        for joint_index in range(p.getNumJoints(self.ghost_arm)):
            if p.getJointInfo(self.ghost_arm, joint_index)[2] != p.JOINT_FIXED:
                joint_info = p.getJointInfo(self.ghost_arm, joint_index)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                random_pos = np.random.uniform(lower_limit, upper_limit)
                p.resetJointState(self.ghost_arm, joint_index, random_pos)

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Reset agent arm to zero pose
        for joint_index in self.agent_controllable_joints:
            p.resetJointState(self.agent_arm, joint_index, targetValue=0, targetVelocity=0)

        # Set random ghost pose
        self._set_random_ghost_pose()

        # Clear and pre-fill deque
        self.historical_imu_data.clear()
        for _ in range(10):
            self.historical_imu_data.append(np.zeros(3, dtype=np.float32))

        # Get observation
        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
