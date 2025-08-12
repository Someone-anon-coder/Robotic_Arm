import os
from stable_baselines3 import SAC
from environment.arm_env import ArmEnv
from config import TRAIN_CONFIG, ENV_CONFIG

if __name__ == "__main__":
    # Create directories for logs and models
    log_dir = os.path.dirname(TRAIN_CONFIG["tensorboard_log"])
    model_dir = os.path.dirname(ENV_CONFIG["model_save_path"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Instantiate Environment
    env = ArmEnv(render_mode=ENV_CONFIG["render_mode"])

    # Instantiate Agent
    model = SAC(env=env, **TRAIN_CONFIG)

    # Start Training
    model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"])

    # Save Model
    model.save(ENV_CONFIG["model_save_path"])

    # Close Environment
    env.close()
