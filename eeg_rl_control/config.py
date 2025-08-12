# Training Hyperparameters
TRAIN_CONFIG = {
    "policy": "MlpPolicy",
    "total_timesteps": 2_000_000,
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "tensorboard_log": "./logs/sac_arm_tensorboard/",
    "verbose": 1,
}

# Environment & Model Saving Configuration
ENV_CONFIG = {
    "env_id": "ArmEnv-v0",
    "render_mode": "human", # Set to 'none' for faster training
    "model_save_path": "./models/sac_arm_model.zip",
}
