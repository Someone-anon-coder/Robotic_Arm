import gymnasium as gym
import numpy as np
import pybullet as p
import collections
import pybullet_data
import os

class ArmEnv(gym.Env):
    SUCCESS_THRESHOLD = 0.05
    def __init__(self, render_mode='human', include_goal_in_obs=False):
        super().__init__()
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        datapath = pybullet_data.getDataPath()
        urdf_root = os.path.join(os.path.dirname(__file__), '..', 'urdf')
        p.setAdditionalSearchPath(urdf_root)
        self.plane = p.loadURDF(os.path.join(datapath, "plane.urdf"))

        # Load arms
        self.agent_arm = p.loadURDF("robotic_arm.urdf", [0, 0, 0], useFixedBase=True)
        self.ghost_arm = p.loadURDF("glove.urdf", [0, 0, 0], useFixedBase=True)

        self._build_joint_maps()

        # Define Action and Observation Spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32)

        observation_dim = 108
        self.goal_dim = 7 if include_goal_in_obs else 0
        observation_dim = 108 + self.goal_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        self.step_counter = 0
        self.max_steps = 1000 # Max steps per episode
        self.prev_pose_error = None
        self.ghost_start_pose = None
        self.ghost_target_pose = None
        self.ghost_interpolation_speed = 0.05

        self.key_links = ['index_dist_link', 'middle_dist_link', 'ring_dist_link', 'pinky_dist_link', 'thumb_dist_link', 'hand_base_link']

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
        self.agent_link_map = {p.getJointInfo(self.agent_arm, i)[12].decode('utf-8'): i for i in range(p.getNumJoints(self.agent_arm))}
        self.ghost_link_map = {p.getJointInfo(self.ghost_arm, i)[12].decode('utf-8'): i for i in range(p.getNumJoints(self.ghost_arm))}
        self.agent_controllable_joints = [v for k,v in self.agent_joint_map.items() if k.endswith('_joint')]

    def _get_zero_pose(self):
        return [0.0] * p.getNumJoints(self.ghost_arm)

    def _get_random_pose(self):
        random_pose = []
        for joint_index in range(p.getNumJoints(self.ghost_arm)):
            if p.getJointInfo(self.ghost_arm, joint_index)[2] != p.JOINT_FIXED:
                joint_info = p.getJointInfo(self.ghost_arm, joint_index)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                random_pos = np.random.uniform(lower_limit, upper_limit)
                random_pose.append(random_pos)
            else:
                random_pose.append(0.0)
        return random_pose

    def _set_random_ghost_pose(self):
        random_pose = self._get_random_pose()
        for i, joint_pos in enumerate(random_pose):
            p.resetJointState(self.ghost_arm, i, joint_pos)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_counter = 0

        # Reset agent arm to zero pose
        for joint_index in self.agent_controllable_joints:
            p.resetJointState(self.agent_arm, joint_index, targetValue=0, targetVelocity=0)

        # Set the ghost arm to the default "pledge" pose
        pledge_pose = self._get_zero_pose()
        for i, joint_pos in enumerate(pledge_pose):
            p.resetJointState(self.ghost_arm, i, joint_pos)

        # Generate a random valid pose and store it as the target
        self.ghost_target_pose = self._get_random_pose()

        self.prev_pose_error = self._get_pose_error()

        # Clear and pre-fill deque
        self.historical_imu_data.clear()
        for _ in range(10):
            self.historical_imu_data.append(np.zeros(3, dtype=np.float32))

        # Get observation
        observation = self._get_observation()

        return observation, {}

    def _get_pose_error(self):
        total_error = 0.0
        for link_name in self.key_links:
            agent_link_index = self.agent_link_map[link_name]
            ghost_link_index = self.ghost_link_map[link_name]

            agent_link_state = p.getLinkState(self.agent_arm, agent_link_index)
            ghost_link_state = p.getLinkState(self.ghost_arm, ghost_link_index)

            agent_pos = np.array(agent_link_state[0])
            ghost_pos = np.array(ghost_link_state[0])

            error = np.linalg.norm(agent_pos - ghost_pos)
            total_error += error
        return total_error

    def _calculate_reward(self):
        pose_error = self._get_pose_error()

        reward = -pose_error
        is_success = False

        if self.step_counter > 50 and pose_error < self.SUCCESS_THRESHOLD:
            reward += 500
            is_success = True

        reward -= 0.1

        if self.prev_pose_error is not None and pose_error > self.prev_pose_error:
            reward -= 0.5

        self.prev_pose_error = pose_error
        self.reward_info = {'is_success': is_success, 'pose_error': pose_error}

        return reward

    def _update_ghost_pose(self):
        current_joint_states = p.getJointStates(self.ghost_arm, range(p.getNumJoints(self.ghost_arm)))
        current_joint_positions = [state[0] for state in current_joint_states]

        for i in range(p.getNumJoints(self.ghost_arm)):
            if p.getJointInfo(self.ghost_arm, i)[2] != p.JOINT_FIXED:
                current_pos = current_joint_positions[i]
                target_pos = self.ghost_target_pose[i]
                new_pos = current_pos + (target_pos - current_pos) * self.ghost_interpolation_speed
                p.resetJointState(self.ghost_arm, i, new_pos)

    def step(self, action):
        self._update_ghost_pose()
        self.step_counter += 1
        MAX_VELOCITY = 1.5 # rad/s

        for i, joint_index in enumerate(self.agent_controllable_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.agent_arm,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=action[i] * MAX_VELOCITY,
                force=100
            )

        p.stepSimulation()

        reward = self._calculate_reward()
        observation = self._get_observation()

        terminated = self.reward_info['is_success']
        truncated = self.step_counter >= self.max_steps

        return observation, reward, terminated, truncated, self.reward_info

    def render(self):
        pass

    def close(self):
        p.disconnect()
