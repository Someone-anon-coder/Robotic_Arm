# eeg_robotic_arm_rl/visualize_setup.py

import pybullet as p
import time
from config.simulation_config import SimConfig
from utils.simulation_utils import setup_simulation, print_joint_info

def main():
    """
    Main function to run the simulation setup and visualization.
    """
    client, robot_id, glove_id = setup_simulation(SimConfig, connect_mode=p.GUI)
    
    print("Simulation setup complete.")
    print(f"Robotic Arm ID: {robot_id}")
    print(f"Glove (Ghost) ID: {glove_id}")
    
    print_joint_info(robot_id, "Robotic Arm")
    print_joint_info(glove_id, "Glove")
    
    print("\nVisualization running. Close the window to exit.")
    
    try:
        while True:
            # This loop is necessary to keep the GUI responsive
            p.stepSimulation()
            time.sleep(SimConfig.SIM_TIME_STEP)
    except p.error as e:
        print(f"PyBullet error: {e}. Exiting.")
    finally:
        if p.isConnected(client):
            p.disconnect(client)
            print("Simulation disconnected.")

if __name__ == "__main__":
    main()