from environment.arm_env import ArmEnv
from stable_baselines3 import SAC
from config import ENV_CONFIG

def main():
    # Instantiate the environment
    env = ArmEnv(render_mode='human')

    # Load the trained model
    model = SAC.load(ENV_CONFIG["model_save_path"])

    # Run evaluation loop
    for episode in range(5):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Get only first 5 and last 5 rewards in each episode
            if env.step_counter < 2 or env.step_counter >= env.max_steps - 2:
                print(f"Step: {env.step_counter}, Reward: {reward}")
            
        print(f"Episode {episode + 1} finished with reward: {reward}")

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()
