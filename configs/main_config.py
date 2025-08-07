# configs/main_config.py

import pathlib

# --- Path Definitions ---
# Get the absolute path to the project's root directory
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Define the path to the URDF files directory
URDF_ROOT = PROJECT_ROOT / "urdf"
ROBOT_URDF_PATH = URDF_ROOT / "robotic_arm.urdf"
GLOVE_URDF_PATH = URDF_ROOT / "glove.urdf"

# --- Simulation Parameters ---
# Time step for the physics simulation
# A smaller time step makes the simulation more accurate but slower
SIMULATION_TIME_STEP = 1. / 240.