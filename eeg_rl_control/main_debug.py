import pybullet as p
import os
import time
import numpy as np
from simulation.environment import BiomimeticArmEnv

def main():
    # Construct the path to the config file relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config/sim_params.yaml')

    # 1. Initialize the Environment
    env = BiomimeticArmEnv(config_path)
    print("Environment Loaded Successfully")

    # 2. Create Debug Sliders
    sliders = {}
    for name, index in env.joint_map_ghost.items():
        info = p.getJointInfo(env.ghost_id, index)
        lower_limit = info[8]
        upper_limit = info[9]
        start_pos = 0

        if start_pos < lower_limit: start_pos = lower_limit
        if start_pos > upper_limit: start_pos = upper_limit

        sliders[name] = p.addUserDebugParameter(name, lower_limit, upper_limit, start_pos)

    # 3. Simulation Loop
    step_count = 0
    try:
        while True:
            # Read slider values and construct the joint_targets dictionary
            joint_targets = {}
            for name, slider_id in sliders.items():
                joint_targets[name] = p.readUserDebugParameter(slider_id)

            # Pass targets to the environment step and get observation
            obs = env.step(joint_targets)
            step_count += 1

            # Print debug info every 50 steps
            if step_count % 50 == 0:
                # IMU Error (convert quaternion to Euler for readability)
                imu_error_quat = obs['agent_wrist_shoulder']['imu_orientation_error_quat']
                imu_error_euler = p.getEulerFromQuaternion(imu_error_quat)
                
                # Index Finger Flex Error
                index_flex_error = obs['agent_fingers_imrp']['index_flex_error']
                
                print("--- SENSOR DEBUG ---")
                print(f"Step: {step_count}")
                print(f"IMU Error (Roll, Pitch, Yaw): "
                      f"({np.rad2deg(imu_error_euler[0]):.2f}, "
                      f"{np.rad2deg(imu_error_euler[1]):.2f}, "
                      f"{np.rad2deg(imu_error_euler[2]):.2f}) degrees")
                print(f"Index Finger Flex Error: {index_flex_error:.4f} meters")
                print("--------------------")


            # The sleep is handled by p.setTimeStep and p.stepSimulation()
            # but a small sleep can prevent CPU spinning if sim is not real-time
            time.sleep(1./240.)

    except p.error as e:
        print(f"PyBullet simulation ended: {e}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()
