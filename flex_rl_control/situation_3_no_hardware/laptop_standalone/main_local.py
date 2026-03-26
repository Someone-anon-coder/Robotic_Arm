# File: flex_rl_control/situation_3_no_hardware/laptop_standalone/main_local.py

import time
import sys

# Import the local modules we just created
from dummy_sensor_generator import DummySensorGenerator
from rl_model_placeholder import DummyRLAgent
from pybullet_sim import PyBulletSim

def main():
    print("Initializing Situation 3: No Hardware / Pure Simulation...")
    
    try:
        # Initialize the three core components
        sensor_gen = DummySensorGenerator()
        rl_agent = DummyRLAgent()
        sim = PyBulletSim()
        print("Simulation initialized successfully.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure you are running this script from the 'flex_rl_control' root directory!")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize: {e}")
        sys.exit(1)

    # Control loop frequency (e.g., 60 Hz)
    update_rate = 1.0 / 60.0

    print("\nStarting control loop. Press Ctrl+C to exit.")
    
    try:
        while True:
            loop_start = time.time()

            # 1. Get fake sensor data (simulate ESP32 input)
            sensor_data = sensor_gen.get_sensor_data()

            # 2. Update the transparent "Ghost" arm to show the desired user pose
            sim.update_ghost(sensor_data)

            # 3. Pass sensor data to the RL Agent to get motor commands
            motor_commands = rl_agent.predict(sensor_data)

            # 4. Apply the motor commands to the solid "Agent" arm
            sim.update_agent(motor_commands)

            # 5. Step the physics simulation forward
            sim.step()

            # Maintain a stable loop rate
            elapsed = time.time() - loop_start
            sleep_time = update_rate - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        import pybullet as p
        p.disconnect()
        print("Cleaned up and exited.")

if __name__ == "__main__":
    main()