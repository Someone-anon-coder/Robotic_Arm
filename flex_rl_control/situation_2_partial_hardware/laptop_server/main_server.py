# File: flex_rl_control/situation_2_partial_hardware/laptop_server/main_server.py

import time
import sys

# Import local modules
from data_processor import ESP32DataProcessor
from rl_model_placeholder import DummyRLAgent
from pybullet_sim import PyBulletSim

# CHANGE THIS to your laptop's Wi-Fi IP Address (e.g., 192.168.1.100)
# Make sure the ESP32 config.h is sending to this exact IP!
HOST_IP = "0.0.0.0" # 0.0.0.0 listens on all available network interfaces
UDP_PORT = 5005

def main():
    print("Initializing Situation 2: Partial Hardware (ESP32 + Laptop)...")
    
    udp_server = None
    
    try:
        # 1. Start the UDP Server to listen to the ESP32
        udp_server = ESP32DataProcessor(ip=HOST_IP, port=UDP_PORT)
        
        # 2. Initialize Dummy RL and Simulator
        rl_agent = DummyRLAgent()
        sim = PyBulletSim()
        print("Simulation and Network initialized successfully.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run this script from the 'flex_rl_control' root directory!")
        if udp_server: udp_server.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize: {e}")
        if udp_server: udp_server.close()
        sys.exit(1)

    # Simulation loop rate (e.g., 60 Hz)
    update_rate = 1.0 / 60.0

    print(f"\nWaiting for ESP32 glove data on port {UDP_PORT}...")
    print("Press Ctrl+C to exit safely.")
    
    try:
        while True:
            loop_start = time.time()

            # 1. Fetch live data asynchronously from the UDP Server thread
            sensor_data = udp_server.get_sensor_data()

            # 2. Update the transparent "Ghost" arm to show the user's actual hand pose
            sim.update_ghost(sensor_data)

            # 3. Pass live data to the RL Agent to get smoothed motor commands
            motor_commands = rl_agent.predict(sensor_data)

            # 4. Apply commands to the solid "Agent" arm
            sim.update_agent(motor_commands)

            # 5. Step PyBullet
            sim.step()

            # Maintain stable loop timing
            elapsed = time.time() - loop_start
            sleep_time = update_rate - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        # Crucial: Cleanly shut down the background UDP thread!
        if udp_server:
            udp_server.close()
        
        import pybullet as p
        if p.isConnected():
            p.disconnect()
            
        print("Cleaned up and exited.")

if __name__ == "__main__":
    main()