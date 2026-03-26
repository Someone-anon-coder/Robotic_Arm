# File: flex_rl_control/situation_1_full_hardware/laptop_server/main_hub.py

import time
import sys
import socket
import json

# Local modules (You will need to place the standalone copies of these in this folder)
from data_processor import ESP32DataProcessor
from rl_model_placeholder import DummyRLAgent
from pybullet_sim import PyBulletSim

# --- NETWORK CONFIGURATION ---
HOST_IP = "0.0.0.0"        # Listen on all interfaces
UDP_PORT_IN = 5005         # Receiving FROM ESP32

RASPI_IP = "192.168.1.101" # <-- CRITICAL: CHANGE THIS TO YOUR RASPBERRY PI'S IP ADDRESS!
UDP_PORT_OUT = 5006        # Sending TO Raspberry Pi

def main():
    print("Initializing Situation 1: Full Hardware Hub (ESP32 -> Laptop -> Raspi)...")
    
    udp_server = None
    # Setup outbound socket to send data to Raspberry Pi
    raspi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Initialize UDP Server for ESP32
        udp_server = ESP32DataProcessor(ip=HOST_IP, port=UDP_PORT_IN)
        
        # Initialize Simulation and RL
        rl_agent = DummyRLAgent()
        sim = PyBulletSim()
        print("Simulation and Network Hub initialized successfully.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run this script from the 'flex_rl_control' root directory!")
        if udp_server: udp_server.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize: {e}")
        if udp_server: udp_server.close()
        sys.exit(1)

    update_rate = 1.0 / 60.0
    print(f"\n[NETWORK] Waiting for ESP32 glove data on port {UDP_PORT_IN}...")
    print(f"[NETWORK] Routing motor commands to Raspi at {RASPI_IP}:{UDP_PORT_OUT}...")
    print("Press Ctrl+C to exit safely.")
    
    try:
        while True:
            loop_start = time.time()

            # 1. Fetch live ESP32 glove data
            sensor_data = udp_server.get_sensor_data()

            # 2. Update transparent Ghost arm 
            sim.update_ghost(sensor_data)

            # 3. RL Agent predicts servo commands
            motor_commands = rl_agent.predict(sensor_data)

            # 4. Update solid Agent arm in PyBullet
            sim.update_agent(motor_commands)
            sim.step()

            # 5. FORWARD COMMANDS TO RASPBERRY PI
            # Convert dictionary to JSON string and send over UDP
            payload = json.dumps(motor_commands)
            raspi_sock.sendto(payload.encode('utf-8'), (RASPI_IP, UDP_PORT_OUT))

            # Maintain stable loop timing
            elapsed = time.time() - loop_start
            sleep_time = update_rate - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        if udp_server:
            udp_server.close()
        raspi_sock.close()
        
        import pybullet as p
        if p.isConnected():
            p.disconnect()
        print("Cleaned up and exited.")

if __name__ == "__main__":
    main()