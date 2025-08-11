# eeg_robotic_arm_rl/config/simulation_config.py

import os
import pybullet as p
import numpy as np

class SimConfig:
    # --- Paths ---
    URDF_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "urdf")
    ROBOT_URDF_PATH = os.path.join(URDF_ROOT, "robotic_arm.urdf")
    GLOVE_URDF_PATH = os.path.join(URDF_ROOT, "glove.urdf")

    # --- Simulation Parameters ---
    SIM_TIME_STEP = 1. / 240.
    GRAVITY = [0, 0, -9.81]

    # --- Robot and Glove State (Place both at the same origin for training) ---
    START_POS = [0, 0, 0]
    START_ORN = p.getQuaternionFromEuler([0, 0, 0])

    # --- Observation Space Config ---
    IMU_HISTORY_LENGTH = 10 # Number of past poses to store for IMU data

    # --- Reward Function Weights ---
    # These weights determine the importance of each reward component
    W_POSE_MATCH = 1.0          # For matching the joint angles
    W_PROXIMITY = 2.0           # For getting closer to the target
    W_GOAL_ACHIEVED = 500.0     # Large bonus for reaching the goal
    W_JERK = -0.1               # Penalty for jerky movements
    W_STEP = -0.01              # Small penalty for each step to encourage efficiency

    # Hierarchical weights for pose matching
    POSE_WEIGHTS = {
        'shoulder': 5.0,
        'elbow': 4.0,
        'wrist': 3.0,
        'palm': 2.0,
        'thumb': 1.5,
        'finger': 1.0 # Default weight for other joints (index, middle, etc.)
    }

    # --- Termination and Truncation Conditions ---
    MAX_EPISODE_STEPS = 240 * 15  # Truncate after 15 seconds
    GOAL_ACHIEVED_THRESHOLD = 0.2 # Threshold for total position error to consider goal achieved