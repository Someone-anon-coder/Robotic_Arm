# configs/main_config.py

import pathlib

# --- Path Definitions ---
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
URDF_ROOT = PROJECT_ROOT / "urdf"
ROBOT_URDF_PATH = URDF_ROOT / "robotic_arm.urdf"
GLOVE_URDF_PATH = URDF_ROOT / "glove.urdf"

# --- Simulation Parameters ---
SIMULATION_TIME_STEP = 1. / 240.

# --- Sensor Simulation Configuration ---
# This dictionary maps a descriptive sensor name to the link names in the glove.urdf
# The distance between these two links will simulate the flex sensor reading.
FLEX_SENSOR_MAPPING = {
    "ef_index": ("index_flexed_tracker_link", "index_tracker_link"),
    "ef_middle": ("middle_flexed_tracker_link", "middle_tracker_link"),
    "ef_ring": ("ring_flexed_tracker_link", "ring_tracker_link"),
    "ef_pinky": ("pinky_flexed_tracker_link", "pinky_tracker_link"),
    "ef_thumb": ("thumb_flexed_tracker_link", "thumb_tracker_link"),
    # Add other flex sensors as needed, e.g., for adduction/abduction
}

# The names of the links that represent our IMU sensors in the glove.urdf
IMU_A_LINK_NAME = "IMU_A_link_tracker"  # Actual IMU on the hand
IMU_R_LINK_NAME = "IMU_R_link_tracker"  # Reference IMU at the base

# The min/max values for a raw flex sensor reading (for normalization)
FLEX_SENSOR_RANGE = [0, 1023]

CONTROLLED_JOINTS = [
    "shoulder_pan_joint", "shoulder_tilt_joint",
    "elbow_flex_joint", "elbow_rot_joint",
    "wrist_flex_joint", "wrist_abd_joint",
    "palm_flex_joint",
    "index_abd_joint", "index_mcp_flex_joint", "index_pip_joint", "index_dip_joint",
    "middle_abd_joint", "middle_mcp_flex_joint", "middle_pip_joint", "middle_dip_joint",
    "ring_abd_joint", "ring_mcp_flex_joint", "ring_pip_joint", "ring_dip_joint",
    "pinky_abd_joint", "pinky_mcp_flex_joint", "pinky_pip_joint", "pinky_dip_joint",
    "thumb_opp_joint", "thumb_z_rot_joint", "thumb_mcp_flex_joint", "thumb_ip_joint"
]

# Ensure we have 27 joints
assert len(CONTROLLED_JOINTS) == 27, "Expected 27 controllable joints!"