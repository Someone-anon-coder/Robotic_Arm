import numpy as np

class RewardCalculator:
    def __init__(self,
                 linear_error_threshold=0.05,
                 angular_error_threshold=0.1,
                 pose_reward_weight=10.0,
                 step_penalty=-0.01,
                 jerk_penalty_weight=-0.5):
        self.linear_error_threshold = linear_error_threshold
        self.angular_error_threshold = angular_error_threshold
        self.pose_reward_weight = pose_reward_weight
        self.step_penalty = step_penalty
        self.jerk_penalty_weight = jerk_penalty_weight
        self.previous_errors = {}

    def compute_reward(self, obs):
        rewards = {}

        # Wrist/Shoulder Reward
        linear_error = np.linalg.norm(obs['agent_wrist_shoulder']['imu_linear_accel_error'])
        angular_error = np.linalg.norm(obs['agent_wrist_shoulder']['imu_orientation_error_quat'][:3])

        pose_reward = 0
        if linear_error < self.linear_error_threshold and angular_error < self.angular_error_threshold:
            pose_reward = self.pose_reward_weight

        jerk_penalty = 0
        if 'wrist_shoulder' in self.previous_errors:
            prev_linear_error, prev_angular_error = self.previous_errors['wrist_shoulder']
            if linear_error > prev_linear_error or angular_error > prev_angular_error:
                jerk_penalty = self.jerk_penalty_weight

        rewards['wrist_shoulder'] = pose_reward + self.step_penalty + jerk_penalty
        self.previous_errors['wrist_shoulder'] = (linear_error, angular_error)

        # Fingers Reward
        finger_errors = [
            obs['agent_fingers_imrp']['index_flex_error'],
            obs['agent_fingers_imrp']['middle_flex_error'],
            obs['agent_fingers_imrp']['ring_flex_error'],
            obs['agent_fingers_imrp']['pinky_flex_error']
        ]
        avg_finger_error = np.mean(np.abs(finger_errors))

        finger_reward = -avg_finger_error  # Negative reward proportional to error
        rewards['fingers'] = finger_reward + self.step_penalty

        # Thumb/Palm Reward
        thumb_error = np.abs(obs['agent_thumb_palm']['thumb_flex_error'])
        thumb_reward = -thumb_error
        rewards['thumb_palm'] = thumb_reward + self.step_penalty

        return rewards
