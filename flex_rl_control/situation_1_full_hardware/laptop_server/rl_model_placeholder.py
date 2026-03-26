# File: flex_rl_control/situation_1_full_hardware/laptop_server/rl_model_placeholder.py

class DummyRLAgent:
    """
    Smoothing and Mapping Agent for Situation 1.
    Processes raw glove data into stable servo targets.
    """
    def __init__(self):
        self.alpha = 0.25 # Smoothing coefficient
        self.prev = {
            "thumb_angle": 0.0, "index_angle": 0.0, "middle_angle": 0.0,
            "ring_angle": 0.0, "pinky_angle": 0.0, "wrist_angle": 0.0
        }

    def predict(self, sensor_data):
        """
        Input: Dict of raw sensor values.
        Output: Dict of smoothed motor angles in radians.
        """
        def to_rad(val):
            return (max(0, min(1023, val)) / 1023.0) * 1.57

        raw = {
            "thumb_angle":  to_rad(sensor_data.get("flex_thumb", 0)),
            "index_angle":  to_rad(sensor_data.get("flex_index", 0)),
            "middle_angle": to_rad(sensor_data.get("flex_middle", 0)),
            "ring_angle":   to_rad(sensor_data.get("flex_ring", 0)),
            "pinky_angle":  to_rad(sensor_data.get("flex_pinky", 0)),
            "wrist_angle":  sensor_data.get("imu_euler", [0,0,0])[1]
        }

        # Apply Exponential Moving Average smoothing
        for k in self.prev:
            self.prev[k] = (self.alpha * raw[k]) + ((1.0 - self.alpha) * self.prev[k])
            
        return self.prev