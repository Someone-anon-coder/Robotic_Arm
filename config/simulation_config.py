# eeg_robotic_arm_rl/config/simulation_config.py

import os
import pybullet as p

class SimConfig:
    # --- Paths ---
    URDF_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "urdf")
    ROBOT_URDF_PATH = os.path.join(URDF_ROOT, "robotic_arm.urdf")
    GLOVE_URDF_PATH = os.path.join(URDF_ROOT, "glove.urdf")

    # --- Simulation Parameters ---
    SIM_TIME_STEP = 1. / 240.
    GRAVITY = [0, 0, -9.81]

    # --- Robot Initial State ---
    # The agent-controlled arm
    ROBOT_START_POS = [0, 0, 0]
    ROBOT_START_ORN = p.getQuaternionFromEuler([0, 0, 0])
    
    # --- Glove (Ghost) Initial State ---
    # The target pose arm, offset for visualization
    GLOVE_START_POS = [0, 0, 0]
    GLOVE_START_ORN = p.getQuaternionFromEuler([0, 0, 0])