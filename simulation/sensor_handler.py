# simulation/sensor_handler.py

import pybullet as p
import numpy as np

class SensorHandler:
    """
    Manages and simulates sensor data from the 'ghost' glove model.
    This includes flex sensors (calculated via distances) and IMUs.
    """
    def __init__(self, glove_model_id, config):
        """
        Initializes the sensor handler.

        Args:
            glove_model_id (int): The PyBullet ID for the loaded glove model.
            config: The configuration module with sensor mappings.
        """
        self.glove_id = glove_model_id
        self.config = config
        self.link_name_to_index = self._get_link_indices()
        
        # Store the indices for the IMU links for quick access
        self.imu_a_index = self.link_name_to_index[self.config.IMU_A_LINK_NAME]
        self.imu_r_index = self.link_name_to_index[self.config.IMU_R_LINK_NAME]
        
        # Prepare a dictionary to hold link pairs for flex sensors
        self.flex_sensor_links = self._setup_flex_sensors()
        
        # Store debug item IDs to be able to remove them later
        self.debug_lines = []

        print("--- Sensor Handler Initialized ---")

    def _get_link_indices(self):
        """Creates a mapping from link names to their PyBullet index."""
        link_map = {}
        for i in range(p.getNumJoints(self.glove_id)):
            joint_info = p.getJointInfo(self.glove_id, i)
            link_name = joint_info[12].decode('UTF-8')
            link_map[link_name] = i
        return link_map

    def _setup_flex_sensors(self):
        """
        Uses the config mapping to get the link indices for each flex sensor pair.
        """
        flex_links = {}
        for sensor_name, link_names in self.config.FLEX_SENSOR_MAPPING.items():
            link1_name, link2_name = link_names
            if link1_name in self.link_name_to_index and link2_name in self.link_name_to_index:
                flex_links[sensor_name] = (
                    self.link_name_to_index[link1_name],
                    self.link_name_to_index[link2_name]
                )
            else:
                print(f"Warning: Could not find links for sensor '{sensor_name}'")
        return flex_links

    def get_flex_sensor_values(self, visualize=False):
        """
        Calculates the distance between sensor link pairs to simulate flex sensor readings.

        Args:
            visualize (bool): If True, draws debug lines in the simulation.

        Returns:
            dict: A dictionary of sensor names and their calculated raw distance values.
        """
        readings = {}
        # Remove previous debug lines if they exist
        if visualize:
            self._remove_debug_items()
            
        for sensor_name, link_indices in self.flex_sensor_links.items():
            link1_index, link2_index = link_indices
            
            # Get the world position of each link
            link1_state = p.getLinkState(self.glove_id, link1_index)
            link2_state = p.getLinkState(self.glove_id, link2_index)
            pos1 = np.array(link1_state[0])
            pos2 = np.array(link2_state[0])
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(pos1 - pos2)
            readings[sensor_name] = distance
            
            # Draw a debug line to visualize the sensor
            if visualize:
                line_id = p.addUserDebugLine(pos1, pos2, lineColorRGB=[0, 1, 0], lineWidth=2)
                self.debug_lines.append(line_id)

        return readings

    def get_imu_values(self):
        """
        Reads the state of the IMU links to get linear/angular velocity and orientation.

        Returns:
            dict: A dictionary containing the data for the actual and reference IMUs.
        """
        imu_a_state = p.getLinkState(self.glove_id, self.imu_a_index, computeLinkVelocity=1)
        imu_r_state = p.getLinkState(self.glove_id, self.imu_r_index, computeLinkVelocity=1)
        
        imu_data = {
            "IMU_a": {
                "linear_velocity": imu_a_state[6],
                "angular_velocity": imu_a_state[7],
                "orientation_quaternion": imu_a_state[1]
            },
            "IMU_r": {
                "linear_velocity": imu_r_state[6],
                "angular_velocity": imu_r_state[7],
                "orientation_quaternion": imu_r_state[1]
            }
        }
        return imu_data

    def _remove_debug_items(self):
        """Removes all currently active debug lines created by this handler."""
        for item_id in self.debug_lines:
            p.removeUserDebugItem(item_id)
        self.debug_lines.clear()