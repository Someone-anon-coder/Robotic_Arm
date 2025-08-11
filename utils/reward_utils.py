# eeg_robotic_arm_rl/utils/reward_utils.py

import numpy as np

def get_weighted_pose_error(robot_pos, glove_pos, joint_names, config):
    """
    Calculates the pose matching error, applying higher weights to more
    critical joints like the shoulder and elbow.
    """
    error = 0.0
    weights_applied = []

    for i, joint_name in enumerate(joint_names):
        weight = config.POSE_WEIGHTS.get('finger', 1.0) # Default weight
        if 'shoulder' in joint_name:
            weight = config.POSE_WEIGHTS['shoulder']
        elif 'elbow' in joint_name:
            weight = config.POSE_WEIGHTS['elbow']
        elif 'wrist' in joint_name:
            weight = config.POSE_WEIGHTS['wrist']
        elif 'palm' in joint_name:
            weight = config.POSE_WEIGHTS['palm']
        elif 'thumb' in joint_name:
            weight = config.POSE_WEIGHTS['thumb']
        
        error += weight * ((robot_pos[i] - glove_pos[i]) ** 2)
        weights_applied.append(weight)

    return np.sqrt(error)

def calculate_reward(
    robot_joint_pos,
    glove_joint_pos,
    robot_joint_vel,
    prev_total_error,
    joint_names,
    config
):
    """
    Calculates the composite reward based on multiple criteria.
    """
    # 1. Pose Matching Reward (weighted)
    total_error = get_weighted_pose_error(robot_joint_pos, glove_joint_pos, joint_names, config)
    pose_reward = -config.W_POSE_MATCH * total_error

    # 2. Proximity Reward (getting closer)
    # Give a reward if the agent has reduced the error from the last step
    proximity_reward = 0
    if total_error < prev_total_error:
        proximity_reward = config.W_PROXIMITY * (prev_total_error - total_error)

    # 3. Jerkiness Penalty
    jerk_penalty = config.W_JERK * np.sum(np.square(robot_joint_vel))

    # 4. Step Penalty
    step_penalty = config.W_STEP

    # --- Total Reward ---
    total_reward = pose_reward + proximity_reward + jerk_penalty + step_penalty
    
    return total_reward, total_error