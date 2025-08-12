import os
from stable_baselines3 import SAC
from environment.arm_env import ArmEnv
from config import SAC_CONFIG, TRAIN_CONFIG, ENV_CONFIG

if __name__ == "__main__":
    # Create directories for logs and models
    log_dir = os.path.dirname(SAC_CONFIG["tensorboard_log"])
    model_dir = os.path.dirname(ENV_CONFIG["model_save_path"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Instantiate Environment
    env = ArmEnv(render_mode=ENV_CONFIG["render_mode"])

    # Instantiate Agent
    model = SAC(env=env, **SAC_CONFIG)

    # Start Training
    model.learn(**TRAIN_CONFIG)

    # Save Model
    model.save(ENV_CONFIG["model_save_path"])

    # Close Environment
    env.close()
