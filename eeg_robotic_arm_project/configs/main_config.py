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
ROBOT_START_POS = [0, 0, 0]
GLOVE_START_POS = [0, 0, 0]
START_ORIENTATION = [0, 0, 0, 1] # As a quaternion [x, y, z, w]

# Define Sensor Ranges: Add constants for the flex sensor output range.
FLEX_SENSOR_MIN = 0
FLEX_SENSOR_MAX = 1023
# We need to estimate the min/max possible distance for a flex sensor to normalize the value.
# For now, let's use an estimated range. We can refine this later.
FLEX_DISTANCE_MIN = 0.0  # When tracker is at the base
FLEX_DISTANCE_MAX = 0.08 # An estimated max distance for a fully flexed finger

# Define Link Names for Sensors: Create dictionaries that map a logical sensor name to the precise link names from glove.urdf. This is critical for making our SensorHandler code clean and readable.
# Names of the links for IMU data
IMU_LINKS = {
    'actual': 'IMU_A_link_tracker',
    'reference': 'IMU_R_link_tracker'
}

# Mapping for flex sensors. Each key is a sensor, and the value is a tuple
# containing the (end_point_link, base_point_link).
FLEX_SENSOR_LINKS = {
    # Finger Flexion/Extension
    'ef_index': ('index_tracker_link', 'index_flexed_tracker_link'),
    'ef_middle': ('middle_tracker_link', 'middle_flexed_tracker_link'),
    'ef_ring': ('ring_tracker_link', 'ring_flexed_tracker_link'),
    'ef_pinky': ('pinky_tracker_link', 'pinky_flexed_tracker_link'),
    'ef_thumb': ('thumb_tracker_link', 'thumb_flexed_tracker_link'),

    # NOTE: For adduction/abduction and other movements, we will simulate them
    # using joint states directly for now, as distance measurement is less intuitive
    # for these. We will focus the SensorHandler on flexion and IMUs first.
}
