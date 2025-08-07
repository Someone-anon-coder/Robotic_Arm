# run_step2_sensors.py

from simulation.environment import RoboticArmEnv
import configs.main_config as cfg
import time
import os

import pybullet as p

def format_sensor_data_for_printing(flex_data, imu_data):
    """Formats the raw sensor data into a readable string."""
    # Use ANSI escape codes for colors
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'

    # Clear the console screen
    os.system('cls' if os.name == 'nt' else 'clear')

    output = f"{HEADER}--- LIVE SENSOR DATA ---{ENDC}\n\n"
    
    output += f"{BLUE}Flex Sensors (Raw Distances):{ENDC}\n"
    for name, value in flex_data.items():
        output += f"  {CYAN}{name:<10}{ENDC}: {GREEN}{value:.4f}{ENDC}\n"
        
    output += f"\n{BLUE}IMU Data:{ENDC}\n"
    
    # Actual IMU (IMU_a)
    imu_a_vel = imu_data['IMU_a']['linear_velocity']
    imu_a_quat = imu_data['IMU_a']['orientation_quaternion']
    output += f"  {CYAN}IMU_a (Hand):{ENDC}\n"
    output += f"    {GREEN}LinVel (x,y,z):{ENDC} ({imu_a_vel[0]:.3f}, {imu_a_vel[1]:.3f}, {imu_a_vel[2]:.3f})\n"
    output += f"    {GREEN}Quat (x,y,z,w):{ENDC} ({imu_a_quat[0]:.3f}, {imu_a_quat[1]:.3f}, {imu_a_quat[2]:.3f}, {imu_a_quat[3]:.3f})\n"

    print(output)
    return output


def main():
    """
    Initializes the environment and runs a loop to test the sensor handler.
    """
    print("--- Starting Step 2: Sensor Simulation and Visualization ---")
    env = None
    log_info = ""
    try:
        env = RoboticArmEnv(config=cfg)
        
        # Add a simple debug slider to move one of the ghost's fingers
        finger_joint_index = env.sensor_handler.link_name_to_index["index_prox_link"]
        joint_info = p.getJointInfo(env.glove_id, finger_joint_index)
        joint_limit_low = joint_info[8]
        joint_limit_high = joint_info[9]
        
        slider = p.addUserDebugParameter(
            "Ghost Index Finger", 
            joint_limit_low, 
            joint_limit_high, 
            0
        )
        
        print("\nStarting simulation loop. Move the slider to see sensor values change.")
        print("Green lines visualize the flex sensors. Close the window to exit.")

        while True:
            # Read the slider value and apply it to the ghost model's joint
            target_pos = p.readUserDebugParameter(slider)
            p.setJointMotorControl2(
                env.glove_id,
                finger_joint_index,
                p.POSITION_CONTROL,
                targetPosition=target_pos
            )
            
            # Get and print sensor data
            flex_data, imu_data = env.get_all_sensor_readings(visualize_flex=True)
            log_info = log_info + "\n\n" + format_sensor_data_for_printing(flex_data, imu_data)

            p.stepSimulation()
            time.sleep(env.config.SIMULATION_TIME_STEP)

    except p.error:
        print("PyBullet window closed by user.")
    finally:
        if env:
            env.close()
        print("--- Step 2 execution finished ---")

        # Remove color codes for logging
        log_info = log_info.replace('\033[95m', '').replace('\033[94m', '').replace('\033[96m', '').replace('\033[92m', '').replace('\033[0m', '')
        with open("sensor_log.txt", "w") as log_file:
            log_file.write(log_info)
    
        print("Sensor data logged to sensor_log.txt")

if __name__ == "__main__":
    main()