# File: flex_rl_control/situation_3_no_hardware/laptop_standalone/rl_model_placeholder.py

class DummyRLAgent:
    """
    Placeholder for the future Reinforcement Learning Model.
    Currently uses simple linear mapping to bypass the need for a trained neural net.
    """
    def __init__(self):
        # Here you would normally load your saved model weights, e.g.:
        # self.model = SAC.load("models/sac_agent_fingers")
        pass

    def predict(self, sensor_data):
        """
        Takes raw sensor data (dict) and outputs desired joint angles (in radians).
        """
        # Helper function: Map flex sensors (0 - 1023) to Servo Angles (0.0 to 1.57 radians)
        def map_flex_to_rad(val):
            # Clamp value just in case
            val = max(0, min(1023, val))
            # 0 -> 0 radians (straight), 1023 -> 1.57 radians (90 degree bend)
            return (val / 1023.0) * 1.57

        # Generate motor commands based on mapped sensor data
        motor_commands = {
            "thumb_angle": map_flex_to_rad(sensor_data["flex_thumb"]),
            "index_angle": map_flex_to_rad(sensor_data["flex_index"]),
            "middle_angle": map_flex_to_rad(sensor_data["flex_middle"]),
            "ring_angle": map_flex_to_rad(sensor_data["flex_ring"]),
            "pinky_angle": map_flex_to_rad(sensor_data["flex_pinky"]),
            
            # Map IMU pitch directly to the wrist tilt servo
            "wrist_angle": sensor_data["imu_euler"][1] 
        }

        return motor_commands