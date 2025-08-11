# eeg_robotic_arm_rl/utils/sensor_utils.py

import pybullet as p
import numpy as np

def get_link_indices_by_name(robot_id, link_names):
    """
    Finds the index for each link specified by name.
    """
    name_to_index = {}
    for i in range(p.getNumJoints(robot_id)):
        link_name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
        if link_name in link_names:
            name_to_index[link_name] = i
    return name_to_index

def get_simulated_flex_sensors(glove_id, link_map):
    """
    Calculates the euclidean distance between tracker and flexed points
    to simulate flex sensor values.
    """
    flex_values = []
    
    finger_prefixes = ["index", "middle", "ring", "pinky", "thumb"]
    
    for prefix in finger_prefixes:
        tracker_link_name = f"{prefix}_tracker_link"
        flexed_link_name = f"{prefix}_flexed_tracker_link"
        
        tracker_idx = link_map.get(tracker_link_name)
        flexed_idx = link_map.get(flexed_link_name)

        if tracker_idx is not None and flexed_idx is not None:
            tracker_state = p.getLinkState(glove_id, tracker_idx)
            flexed_state = p.getLinkState(glove_id, flexed_idx)
            
            tracker_pos = np.array(tracker_state[0])
            flexed_pos = np.array(flexed_state[0])
            
            distance = np.linalg.norm(tracker_pos - flexed_pos)
            
            # Simple mapping, can be refined later
            # We assume smaller distance means more flex (higher value)
            # This is an inverse relationship and needs calibration, but for now:
            simulated_value = 1023 * (1 - min(distance / 0.1, 1.0)) # Normalize based on an assumed max distance
            flex_values.append(simulated_value)

    # We need to ensure we have a consistent number of outputs.
    # The prompt implies more flex sensors. This is a simplified placeholder.
    # We will pad to 14 for now, as specified in the original prompt.
    while len(flex_values) < 14:
        flex_values.append(0.0)
        
    return np.array(flex_values, dtype=np.float32)