# Centralizes all hyperparameters, URDF paths, joint names, and other settings to avoid magic numbers.
import os

# Get the absolute path of the project's root directory.
# This assumes the script is run from the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create variables storing the absolute paths to the URDF files.
# Use Python's `os` or `pathlib` module to ensure cross-platform compatibility.
ROBOT_URDF_PATH = os.path.join(PROJECT_ROOT, "urdf", "robotic_arm.urdf")
GLOVE_URDF_PATH = os.path.join(PROJECT_ROOT, "urdf", "glove.urdf")

# Define the initial base position and orientation for both the robot and the glove.
# For this visualization step, let's place them slightly apart on the Y-axis so we can see both clearly.
ROBOT_START_POS = [0, -0.5, 0]
GLOVE_START_POS = [0, 0.5, 0]
START_ORIENTATION = [0, 0, 0, 1] # As a quaternion [x, y, z, w]
