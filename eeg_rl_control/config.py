# SAC Model Configuration
SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 200_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "tensorboard_log": "./logs/sac_arm_tensorboard/",
    "verbose": 1,
}

# Training Loop Configuration
TRAIN_CONFIG = {
    "total_timesteps": 500_000,
}

# Environment & Model Saving Configuration
ENV_CONFIG = {
    "env_id": "ArmEnv-v0",
    "render_mode": "none", # Set to 'none' for faster training
    "model_save_path": "./models/sac_arm_model.zip",
}
