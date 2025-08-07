# run_step3_imitation.py

from simulation.environment import RoboticArmEnv
import configs.main_config as cfg
import time
import pybullet as p

def main():
    print("--- Starting Step 3: Direct Pose Imitation ---")
    env = None
    try:
        env = RoboticArmEnv(config=cfg)
        observation, info = env.reset()
        
        # --- Create sliders to control the GHOST model ---
        sliders = {}
        joints_to_control = [
            "shoulder_pan_joint", "shoulder_tilt_joint",
            "elbow_flex_joint", "elbow_rot_joint",
            "wrist_abd_joint", "wrist_flex_joint",
            "palm_flex_joint",
            "index_abd_joint", "index_mcp_flex_joint", "index_pip_joint", "index_dip_joint",
            "middle_abd_joint", "middle_mcp_flex_joint", "middle_pip_joint", "middle_dip_joint",
            "ring_abd_joint", "ring_mcp_flex_joint", "ring_pip_joint", "ring_dip_joint",
            "pinky_abd_joint", "pinky_mcp_flex_joint", "pinky_pip_joint", "pinky_dip_joint",
            "thumb_opp_joint", "thumb_z_rot_joint", "thumb_mcp_flex_joint", "thumb_ip_joint"
        ]
        
        for joint_name in joints_to_control:
            joint_index = env.glove_joint_map[joint_name]
            joint_info = p.getJointInfo(env.glove_id, joint_index)
            low, high = joint_info[8], joint_info[9]
            slider_id = p.addUserDebugParameter(joint_name, low, high, 0)
            sliders[joint_name] = slider_id
            
        print("\nMove sliders to control the transparent ghost arm.")
        print("The solid robotic arm will directly imitate its pose.")
        
        while True:
            # --- 1. Control the Ghost Arm with Sliders ---
            for joint_name, slider_id in sliders.items():
                target_pos = p.readUserDebugParameter(slider_id)
                joint_index = env.glove_joint_map[joint_name]
                p.setJointMotorControl2(
                    env.glove_id, joint_index, p.POSITION_CONTROL, target_pos
                )
            
            # --- 2. Get the Ghost's Pose to use as the Action ---
            # This is our "perfect" action in this imitation scenario
            action = env.get_model_joint_states(env.glove_id)
            
            # --- 3. Step the Environment with the Perfect Action ---
            # This applies the action to the ROBOTIC arm
            observation, reward, _, _, _ = env.step(action)
            
            # Optional: Print the reward to see how well it's tracking
            print(f"Tracking Reward (neg MSE): {reward:.6f}", end='\r')

            time.sleep(env.config.SIMULATION_TIME_STEP)

    except p.error:
        print("\nPyBullet window closed by user.")
    finally:
        if env:
            env.close()
    print("\n--- Step 3 execution finished ---")

if __name__ == "__main__":
    main()