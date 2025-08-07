"""
Central configuration file for the project.

Contains all hyperparameters, file paths, joint names, and other
constants to ensure consistency and ease of modification.
"""
# Simulation Parameters
SIM_TIMESTEP = 1.0 / 240.0
SIM_GRAVITY = -9.81
# URDF File Paths
ROBOT_URDF_PATH = "urdf/robotic_arm.urdf"
GLOVE_URDF_PATH = "urdf/glove.urdf"
# RL Hyperparameters
# TODO: Add learning rates, gamma, tau, batch size, etc.
# Joint and Link Names
# TODO: Add lists of joint names for easier access