# File: flex_rl_control/situation_3_no_hardware/laptop_standalone/dummy_sensor_generator.py

import time
import math

class DummySensorGenerator:
    """
    Generates fake sensor data to simulate the ESP32 and glove hardware.
    Creates oscillating sine waves to mimic a human opening/closing their hand.
    """
    def __init__(self):
        self.start_time = time.time()

    def get_sensor_data(self):
        """
        Returns a dictionary with fake flex sensor and IMU data.
        Flex range: 0 (flat/extended) to 1023 (fully bent/flexed)
        IMU range: Euler angles in radians
        """
        # Time variable for oscillation
        t = time.time() - self.start_time

        # Oscillate flex sensors between 0 and 1023
        # math.sin(t) goes from -1 to 1. We shift and scale it to 0.0 to 1.0
        flex_base = (math.sin(t * 1.5) + 1.0) / 2.0 
        flex_val = int(flex_base * 1023)

        # Oscillate IMU slightly (e.g., wrist bending back and forth)
        roll = 0.0
        pitch = math.sin(t * 0.8) * 0.8 # Oscillates between -0.8 and 0.8 radians
        yaw = 0.0

        return {
            "flex_thumb": flex_val,
            "flex_index": flex_val,
            "flex_middle": flex_val,
            "flex_ring": flex_val,
            "flex_pinky": flex_val,
            "imu_euler": [roll, pitch, yaw]
        }