# File: flex_rl_control/situation_1_full_hardware/raspi_arm_client/main_raspi.py

import socket
import json
import time
from hardware_servo import ArmHardwareController

# Listen on all network interfaces
UDP_IP = "0.0.0.0" 
# Port to receive data FROM the Laptop
UDP_PORT = 5006     

def main():
    print("Starting Raspberry Pi Arm Client (Receiver)...")
    
    # Initialize the physical servo controller
    controller = ArmHardwareController()
    
    # Setup UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[NETWORK] Listening for RL motor commands on port {UDP_PORT}...")
    
    try:
        while True:
            # Receive network packet from Laptop
            data, addr = sock.recvfrom(2048)
            payload = data.decode('utf-8')
            
            try:
                # Parse the target angles
                motor_commands = json.loads(payload)
                
                # Command the physical servos
                controller.update_servos(motor_commands)
                
            except json.JSONDecodeError:
                print("[WARNING] Received malformed JSON packet.")
            except Exception as e:
                print(f"[ERROR] Motor update failed: {e}")
                
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down Raspberry Pi Arm Client safely.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()