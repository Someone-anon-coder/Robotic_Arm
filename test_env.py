# eeg_robotic_arm_rl/test_environment.py

import gymnasium as gym
import numpy as np
import time
from envs.robotic_arm_env import RoboticArmEnv

def main():
    """
    Tests the custom RoboticArmEnv.
    """
    # Instantiate the environment
    env = RoboticArmEnv(render_mode='human')
    
    print("--- Environment Test ---")
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    
    # Reset the environment
    obs, info = env.reset()
    print("\nInitial Observation Shape:", obs.shape)
    
    # Run a short episode with random actions
    episode_reward = 0
    for step in range(240 * 5): # Run for 5 seconds
        # Take a random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        
        if (step + 1) % 240 == 0: # Print reward every second
            print(f"Step: {step+1}, Current Reward: {reward:.4f}, Episode Reward: {episode_reward:.4f}")
            
        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()
            episode_reward = 0
            
    # Close the environment
    env.close()
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()