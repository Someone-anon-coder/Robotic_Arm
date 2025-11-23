import pybullet as p
import numpy as np

class SensorManager:
    """
    Manages virtual sensors for the robotic arm simulation.
    Handles both physical links (Ghost) and virtual offset calculations (Robot).
    """
    
    # Offsets extracted from glove.urdf to simulate sensors on the robotic arm
    # which lacks the specific tracker links.
    # Format: 'tracker_link_name': ('parent_link_name', [x, y, z])
    ROBOT_VIRTUAL_OFFSETS = {
        'index_flexed_tracker_link': ('hand_base_link', [0.065, -0.029, 0]),
        'index_tracker_link': ('index_dist_link', [0.02, 0, 0]),
        
        'middle_flexed_tracker_link': ('hand_base_link', [0.065, -0.0084, 0]),
        'middle_tracker_link': ('middle_dist_link', [0.02, 0, 0]),
        
        'ring_flexed_tracker_link': ('hand_base_link', [0.065, 0.01, 0]),
        'ring_tracker_link': ('ring_dist_link', [0.02, 0, 0]),
        
        'pinky_flexed_tracker_link': ('hand_base_link', [0.066, 0.029, 0]),
        'pinky_tracker_link': ('pinky_dist_link', [0.02, 0, 0]),
        
        'thumb_flexed_tracker_link': ('hand_base_link', [0.015, -0.035, 0]),
        'thumb_tracker_link': ('thumb_dist_link', [0.02, 0, 0]),
    }

    def __init__(self, ghost_id, robot_id, sensor_config):
        self.ghost_id = ghost_id
        self.robot_id = robot_id
        self.config = sensor_config

        self.ghost_link_map = self._build_link_map(self.ghost_id)
        self.robot_link_map = self._build_link_map(self.robot_id)

    def _build_link_map(self, body_id):
        """Creates a mapping from link names to their PyBullet indices."""
        link_map = {}
        # -1 is the base link index in PyBullet logic for some functions, 
        # but typically base is handled separately. 
        # However, getLinkState supports -1 for base if we map it manually.
        for i in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, i)
            link_name = info[12].decode('utf-8')
            link_map[link_name] = i
        
        # Map 'base_link' to -1 (PyBullet convention for the root link)
        link_map['base_link'] = -1 
        return link_map

    def _get_link_position(self, body_id, link_map, link_name):
        """
        Robustly gets the world position of a link. 
        If the link exists, returns its position.
        If not (Robot case), calculates position using virtual offsets.
        """
        if link_name in link_map:
            # Direct lookup (Ghost Arm)
            link_index = link_map[link_name]
            if link_index == -1:
                pos, _ = p.getBasePositionAndOrientation(body_id)
            else:
                pos = p.getLinkState(body_id, link_index)[0]
            return np.array(pos)
        
        elif link_name in self.ROBOT_VIRTUAL_OFFSETS:
            # Virtual calculation (Robot Arm)
            parent_name, offset = self.ROBOT_VIRTUAL_OFFSETS[link_name]
            
            if parent_name not in link_map:
                # Fallback for debugging, though URDFs should match structure
                return np.zeros(3)

            parent_index = link_map[parent_name]
            
            if parent_index == -1:
                parent_pos, parent_orn = p.getBasePositionAndOrientation(body_id)
            else:
                link_state = p.getLinkState(body_id, parent_index)
                parent_pos = link_state[0]
                parent_orn = link_state[1]

            # Apply local offset to parent frame to get world position
            # multiplyTransforms(parent_pos, parent_orn, offset, identity_quat)
            new_pos, _ = p.multiplyTransforms(parent_pos, parent_orn, offset, [0, 0, 0, 1])
            return np.array(new_pos)
        
        else:
            raise KeyError(f"Link '{link_name}' not found in body {body_id} or virtual offsets.")

    def get_flex_value(self, body_id, link_map, start_link_name, end_link_name):
        """Calculates the Euclidean distance between two links (Virtual Flex Sensor)."""
        start_pos = self._get_link_position(body_id, link_map, start_link_name)
        end_pos = self._get_link_position(body_id, link_map, end_link_name)
        return np.linalg.norm(start_pos - end_pos)

    def get_imu_data(self, body_id, link_map, link_name):
        """
        Returns IMU data.
        """
        link_index = link_map.get(link_name)
        
        if link_index is None:
             # Fallback or error handling
             return {'orientation': [0,0,0,1], 'linear_acceleration': np.zeros(3), 'angular_velocity': np.zeros(3)}

        if link_index == -1:
            # Base link case
            pos, orn = p.getBasePositionAndOrientation(body_id)
            lin_vel, ang_vel = p.getBaseVelocity(body_id)
        else:
            # computeLinkVelocity=1 to get velocities
            state = p.getLinkState(body_id, link_index, computeLinkVelocity=1)
            orn = state[1]
            lin_vel = state[6]
            ang_vel = state[7]

        return {
            'orientation': orn,
            'linear_acceleration': np.array(lin_vel), # Using velocity as proxy for this phase
            'angular_velocity': np.array(ang_vel)
        }

    def compute_observation(self):
        """
        Computes the full observation dictionary.
        """
        # 1. Get Sensor Data for Ghost (Target)
        ghost_imu = self.get_imu_data(self.ghost_id, self.ghost_link_map, self.config['imu']['actual'])
        ghost_flex = {}
        for name, links in self.config['flex_sensors'].items():
            ghost_flex[name] = self.get_flex_value(self.ghost_id, self.ghost_link_map, links['start'], links['end'])

        # 2. Get Sensor Data for Robot (Current)
        robot_imu = self.get_imu_data(self.robot_id, self.robot_link_map, self.config['imu']['actual'])
        robot_flex = {}
        for name, links in self.config['flex_sensors'].items():
            robot_flex[name] = self.get_flex_value(self.robot_id, self.robot_link_map, links['start'], links['end'])

        # 3. Calculate Errors
        # Orientation Error (Quaternion diff: q_diff = q_ghost^-1 * q_robot)
        # Note: PyBullet quaternions are [x,y,z,w]
        q_ghost_inv = p.invertTransform([0,0,0], ghost_imu['orientation'])[1]
        # Difference to rotate Robot to match Ghost
        orientation_error_q = p.multiplyTransforms([0,0,0], q_ghost_inv, [0,0,0], robot_imu['orientation'])[1]
        
        linear_accel_error = ghost_imu['linear_acceleration'] - robot_imu['linear_acceleration']
        
        flex_errors = {name: ghost_flex[name] - robot_flex[name] for name in ghost_flex}

        # 4. Structure Observation
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
