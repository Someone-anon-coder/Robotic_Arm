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

# Hierarchical RL Configuration
HRL_CONFIG = {
    "total_timesteps": 2_000_000,
    "manager_freq": 10, # Manager acts every 10 steps
    "log_interval": 2048, # Log to tensorboard every 2048 steps
}

MANAGER_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 500_000,
    "batch_size": 256,
    "gamma": 0.99, # Discount factor for future extrinsic rewards
    "policy_kwargs": dict(net_arch=[256, 256]),
    "tensorboard_log": "./logs/hrl/manager/",
    "verbose": 1,
}

CONTROLLER_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "gamma": 0.95, # Discount factor for future intrinsic rewards
    "policy_kwargs": dict(net_arch=[256, 256]),
    "tensorboard_log": "./logs/hrl/controller/",
    "verbose": 0, # Quieter logging for the inner loop agent
}
