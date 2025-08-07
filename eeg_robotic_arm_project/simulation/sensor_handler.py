import pybullet as p
import numpy as np

class SensorHandler:
    def __init__(self, body_id, config):
        """
        Initializes the SensorHandler.
        :param body_id: The PyBullet body unique ID for the glove model.
        :param config: The main configuration file.
        """
        self.body_id = body_id
        self.config = config
        self.link_name_to_index = {}
        self._map_link_names_to_indices()

    def _map_link_names_to_indices(self):
        """
        Maps the link names from the URDF to their corresponding PyBullet joint indices.
        This is crucial for referencing links by name rather than magic numbers.
        """
        num_joints = p.getNumJoints(self.body_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.body_id, i)
            link_name = joint_info[12].decode('utf-8')
            self.link_name_to_index[link_name] = i

    def _get_link_world_position(self, link_name):
        """
        Retrieves the world position of a given link.
        :param link_name: The name of the link.
        :return: A NumPy array representing the link's world position [x, y, z].
        """
        if link_name not in self.link_name_to_index:
            raise KeyError(f"Link '{link_name}' not found in the model's link names.")
        link_index = self.link_name_to_index[link_name]
        link_state = p.getLinkState(self.body_id, link_index)
        return np.array(link_state[0])

    def get_flex_sensor_values(self):
        """
        Calculates the simulated flex sensor values based on the distance
        between specified pairs of links.
        The distance is then normalized to a typical sensor range (e.g., 0-1023).
        :return: A dictionary mapping sensor names to their normalized integer values.
        """
        flex_values = {}
        for name, (end_link, base_link) in self.config.FLEX_SENSOR_LINKS.items():
            end_pos = self._get_link_world_position(end_link)
            base_pos = self._get_link_world_position(base_link)

            distance = np.linalg.norm(end_pos - base_pos)

            # Normalize the distance to the flex sensor's output range
            # Clip the distance to the expected range to avoid out-of-bounds errors
            clipped_dist = np.clip(distance, self.config.FLEX_DISTANCE_MIN, self.config.FLEX_DISTANCE_MAX)

            # Inverted linear interpolation: smaller distance (more flex) -> higher value
            normalized_value = (self.config.FLEX_DISTANCE_MAX - clipped_dist) / \
                               (self.config.FLEX_DISTANCE_MAX - self.config.FLEX_DISTANCE_MIN)

            sensor_value = int(normalized_value * (self.config.FLEX_SENSOR_MAX - self.config.FLEX_SENSOR_MIN) + self.config.FLEX_SENSOR_MIN)

            flex_values[name] = sensor_value

        return flex_values

    def get_imu_values(self):
        """
        Retrieves the kinematic data (position, orientation, velocities) for the IMU links.
        :return: A dictionary containing the data for 'actual' and 'reference' IMUs.
        """
        imu_values = {}
        for name, link_name in self.config.IMU_LINKS.items():
            if link_name not in self.link_name_to_index:
                raise KeyError(f"IMU Link '{link_name}' not found in the model's link names.")

            link_index = self.link_name_to_index[link_name]

            # Get link state with velocity information
            state = p.getLinkState(self.body_id, link_index, computeLinkVelocity=1)

            pos = np.array(state[0])
            orn = np.array(state[1]) # Quaternion
            lin_vel = np.array(state[6])
            ang_vel = np.array(state[7])

            imu_values[name] = {
                'pos': pos,
                'orn': orn,
                'lin_vel': lin_vel,
                'ang_vel': ang_vel
            }

        return imu_values
