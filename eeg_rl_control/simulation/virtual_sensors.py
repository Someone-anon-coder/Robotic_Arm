import pybullet as p
import numpy as np

class SensorManager:
    """
    Manages virtual sensors for the robotic arm simulation.
    It calculates sensor data for both the 'ghost' (target) and 'robot' (actual) models
    and computes the error between them to form an observation for the RL agent.
    """
    def __init__(self, ghost_id, robot_id, sensor_config):
        """
        Initializes the SensorManager.
        Args:
            ghost_id (int): The PyBullet body unique ID for the ghost model.
            robot_id (int): The PyBullet body unique ID for the robot model.
            sensor_config (dict): A dictionary loaded from sensor_config.yaml.
        """
        self.ghost_id = ghost_id
        self.robot_id = robot_id
        self.config = sensor_config

        self.ghost_link_map = self._build_link_map(self.ghost_id)
        self.robot_link_map = self._build_link_map(self.robot_id)

    def _build_link_map(self, body_id):
        """Creates a mapping from link names to their PyBullet indices."""
        link_map = {}
        for i in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, i)
            link_name = info[12].decode('utf-8')
            link_map[link_name] = i
        # Also add the base link
        link_map['base_link'] = -1
        return link_map

    def get_flex_value(self, body_id, link_map, start_link_name, end_link_name):
        """Calculates the Euclidean distance between two links."""
        start_link_index = link_map[start_link_name]
        end_link_index = link_map[end_link_name]

        start_pos = p.getLinkState(body_id, start_link_index)[0]
        end_pos = p.getLinkState(body_id, end_link_index)[0]

        return np.linalg.norm(np.array(start_pos) - np.array(end_pos))

    def get_imu_data(self, body_id, link_map, link_name):
        """
        Returns IMU data for a specific link.
        Note: PyBullet's getLinkState provides linear velocity, not acceleration.
        We use linear velocity as a proxy for linear acceleration.
        """
        link_index = link_map[link_name]
        state = p.getLinkState(body_id, link_index, computeLinkVelocity=1)
        
        # state[1] is worldLinkFrameOrientation (quaternion)
        # state[6] is worldLinkLinearVelocity
        # state[7] is worldLinkAngularVelocity
        return {
            'orientation': state[1],
            'linear_acceleration': np.array(state[6]),
            'angular_velocity': np.array(state[7])
        }

    def compute_observation(self):
        """
        Computes the full observation dictionary by calculating the error
        between the ghost and robot sensor readings.
        """
        # 1. Get Sensor Data for Ghost (The Target)
        ghost_imu = self.get_imu_data(self.ghost_id, self.ghost_link_map, self.config['imu']['actual'])
        ghost_flex = {}
        for name, links in self.config['flex_sensors'].items():
            ghost_flex[name] = self.get_flex_value(self.ghost_id, self.ghost_link_map, links['start'], links['end'])

        # 2. Get Sensor Data for Robot (The Current State)
        robot_imu = self.get_imu_data(self.robot_id, self.robot_link_map, self.config['imu']['actual'])
        robot_flex = {}
        for name, links in self.config['flex_sensors'].items():
            robot_flex[name] = self.get_flex_value(self.robot_id, self.robot_link_map, links['start'], links['end'])

        # 3. Calculate Error (Ghost Value - Robot Value)
        
        # Quaternion difference for orientation error
        q_ghost_inv = p.invertTransform([0,0,0], ghost_imu['orientation'])[1]
        orientation_error_q = p.multiplyTransforms([0,0,0], q_ghost_inv, [0,0,0], robot_imu['orientation'])[1]
        
        linear_accel_error = ghost_imu['linear_acceleration'] - robot_imu['linear_acceleration']
        
        flex_errors = {name: ghost_flex[name] - robot_flex[name] for name in ghost_flex}

        # 4. Return a dictionary structured by Agent
        observation = {
            'agent_wrist_shoulder': {
                'imu_orientation_error_quat': orientation_error_q,
                'imu_linear_accel_error': linear_accel_error
            },
            'agent_fingers_imrp': {
                'index_flex_error': flex_errors['index'],
                'middle_flex_error': flex_errors['middle'],
                'ring_flex_error': flex_errors['ring'],
                'pinky_flex_error': flex_errors['pinky']
            },
            'agent_thumb_palm': {
                'thumb_flex_error': flex_errors['thumb']
            }
        }
        return observation
