import pybullet as p
import os
import time
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
    try:
        while True:
            # Read slider values and construct the joint_targets dictionary
            joint_targets = {}
            for name, slider_id in sliders.items():
                joint_targets[name] = p.readUserDebugParameter(slider_id)

            # Pass targets to the environment step
            env.step(joint_targets)

            # The sleep is handled by p.setTimeStep and p.stepSimulation()
            # but a small sleep can prevent CPU spinning if sim is not real-time
            time.sleep(1./240.)

    except p.error as e:
        print(f"PyBullet simulation ended: {e}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()
