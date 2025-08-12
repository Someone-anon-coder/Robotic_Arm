from environment.arm_env import ArmEnv
from stable_baselines3 import SAC
from config import ENV_CONFIG

def main():
    # Instantiate the environment
    env = ArmEnv(render_mode='human')

    # Load the trained model
    model = SAC.load(ENV_CONFIG["model_save_path"])

    # Run evaluation loop
    for episode in range(10):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
