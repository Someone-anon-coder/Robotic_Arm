"""
Manages the simulation of all sensors on the glove.

This class reads data from the 'glove.urdf' model in the simulation
to generate realistic sensor values for:
- Flex Sensors (calculated via Euclidean distance).
- IMU Sensors (reading link states).
"""
import pybullet as p
import numpy as np

class SensorHandler:
    def __init__(self, glove_model_id, config):
        # TODO: Store model ID and get link/joint indices from config
        self.glove_id = glove_model_id
        self.config = config

    def get_flex_sensor_values(self):
        # TODO: Calculate distances between tracker and flexed links
        # and normalize them to the 0-1023 range.
        return {} # Placeholder

    def get_imu_values(self):
        # TODO: Get world state for IMU_A and IMU_R links.
        return {} # Placeholder
