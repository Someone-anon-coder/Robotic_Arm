"""
This script initializes and runs the ArmVisualizer to display the robotic arm and glove models.
"""
import pybullet as p
from simulation.environment import ArmVisualizer
from configs import main_config

if __name__ == "__main__":
    visualizer = None
    try:
        visualizer = ArmVisualizer(config=main_config)
        visualizer.run()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if visualizer:
            visualizer.close()
