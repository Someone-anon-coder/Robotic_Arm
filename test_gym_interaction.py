import time
from eeg_rl_control.environment.arm_env import ArmEnv

def main():
    env = ArmEnv(render_mode='direct')
    obs, info = env.reset()

    for i in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step: {i}, Reward: {reward:.3f}, Pose Error: {info.get('pose_error', 0):.3f}, Done: {terminated or truncated}")

        if terminated or truncated:
            print("EPISODE FINISHED.")
            obs, info = env.reset()

    env.close()

if __name__ == '__main__':
    main()
