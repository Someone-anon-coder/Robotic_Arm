# File: flex_rl_control/situation_2_partial_hardware/laptop_server/rl_model_placeholder.py

class DummyRLAgent:
    """
    Placeholder for the RL Agent.
    Since real hardware flex sensors are noisy, this dummy agent includes 
    a simple Exponential Moving Average (EMA) filter to smooth the jitter 
    before sending the commands to the robotic arm.
    """
    def __init__(self):
        # Smoothing factor (0.0 to 1.0). Lower = smoother but more delay.
        self.alpha = 0.2 
        
        # State memory for smoothing
        self.smoothed_commands = {
            "thumb_angle": 0.0,
            "index_angle": 0.0,
            "middle_angle": 0.0,
            "ring_angle": 0.0,
            "pinky_angle": 0.0,
            "wrist_angle": 0.0
        }

    def predict(self, sensor_data):
        """
        Takes raw live sensor data (dict) and outputs smoothed joint angles (radians).
        """
        def map_flex_to_rad(val):
            val = max(0, min(1023, val))
            return (val / 1023.0) * 1.57

        # 1. Calculate Raw Targets
        raw_commands = {
            "thumb_angle": map_flex_to_rad(sensor_data.get("flex_thumb", 0)),
            "index_angle": map_flex_to_rad(sensor_data.get("flex_index", 0)),
            "middle_angle": map_flex_to_rad(sensor_data.get("flex_middle", 0)),
            "ring_angle": map_flex_to_rad(sensor_data.get("flex_ring", 0)),
            "pinky_angle": map_flex_to_rad(sensor_data.get("flex_pinky", 0)),
        }
        
        imu_euler = sensor_data.get("imu_euler",[0.0, 0.0, 0.0])
        raw_commands["wrist_angle"] = imu_euler[1] if len(imu_euler) >= 2 else 0.0

        # 2. Apply Exponential Smoothing (RL Agent imitation)
        for key in self.smoothed_commands:
            self.smoothed_commands[key] = (self.alpha * raw_commands[key]) + \
                                          ((1.0 - self.alpha) * self.smoothed_commands[key])

        return self.smoothed_commands