import pybullet as p
import time
import numpy as np
import os
import pybullet_data
# Add the project root to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eeg_robotic_arm_project.configs import main_config as config
from eeg_robotic_arm_project.simulation.sensor_handler import SensorHandler

def main():
    """
    Main function to run the interactive sensor test.
    """
    # --- Setup ---
    client = p.connect(p.DIRECT) # Use DIRECT mode for headless execution
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")

    # Load the glove model
    glove_id = p.loadURDF(config.GLOVE_URDF_PATH, config.GLOVE_START_POS, useFixedBase=True)

    # --- Sensor Handler ---
    sensor_handler = SensorHandler(glove_id, config)

    # --- Create GUI Sliders for Controllable Joints ---
    joint_sliders = {}

    # Define a few key joints to control with sliders
    # We can add more here if needed
    controllable_joints = [
        "wrist_flex_joint",
        "index_mcp_flex_joint",
        "middle_mcp_flex_joint",
        "ring_mcp_flex_joint",
        "pinky_mcp_flex_joint",
        "thumb_mcp_flex_joint"
    ]

    # Dynamically find joint indices and limits
    joint_name_to_index = {p.getJointInfo(glove_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(glove_id))}

    for joint_name in controllable_joints:
        if joint_name in joint_name_to_index:
            joint_index = joint_name_to_index[joint_name]
            joint_info = p.getJointInfo(glove_id, joint_index)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            slider_id = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0)
            joint_sliders[slider_id] = joint_index
        else:
            print(f"Warning: Joint '{joint_name}' not found in URDF.")


    # --- Main Loop ---
    print("Running simulation in DIRECT mode to test sensor logic.")
    try:
        # Programmatically flex a joint to test the sensors
        joint_to_flex = "index_mcp_flex_joint"
        if joint_to_flex in joint_name_to_index:
            joint_index = joint_name_to_index[joint_to_flex]
            p.setJointMotorControl2(
                bodyUniqueId=glove_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=1.5
            )

        # Settle the simulation
        for _ in range(240):
            p.stepSimulation()

        # Get and print the final sensor data
        print("\n--- Final Sensor Readings ---")
        flex_data = sensor_handler.get_flex_sensor_values()
        imu_data = sensor_handler.get_imu_values()

        print("\nFlex Sensors:")
        for sensor, value in flex_data.items():
            print(f"  {sensor}: {value}")

        print("\nIMU (Actual):")
        print(f"  Position: {np.round(imu_data['actual']['pos'], 3)}")
        print(f"  Orientation (Quat): {np.round(imu_data['actual']['orn'], 3)}")

    except p.error as e:
        print(f"PyBullet error: {e}")
    finally:
        p.disconnect()

if __name__ == '__main__':
    main()
