# run_step1_visualization.py

from simulation.environment import RoboticArmEnv
import configs.main_config as cfg

def main():
    """
    Main function to initialize and run the simulation environment for Step 1.
    """
    print("--- Starting Step 1: Environment and Visualization Setup ---")
    
    env = None
    try:
        # Initialize our environment with the configuration
        env = RoboticArmEnv(config=cfg)
        # Run the visualization loop
        env.run_visualization()
    finally:
        # This ensures that we disconnect cleanly even if an error occurs
        if env:
            env.close()
    print("--- Step 1 execution finished ---")

if __name__ == "__main__":
    main()