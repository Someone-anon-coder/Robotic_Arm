import time
from environment.arm_env import ArmEnv

def main():
    env = ArmEnv(render_mode='human')
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    time.sleep(5)
    env.close()

if __name__ == '__main__':
    main()
