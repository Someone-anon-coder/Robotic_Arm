MANAGER_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": (1, "step"),
    "gradient_steps": -1,
    "learning_starts": 10000,
    "policy_kwargs": {"net_arch": [128, 128]},
}

CONTROLLER_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": (1, "step"),
    "gradient_steps": -1,
    "learning_starts": 10000,
    "policy_kwargs": {"net_arch": [256, 256]},
}

HRL_TRAIN_CONFIG = {
    "total_timesteps": 1_000_000,
    "manager_update_freq_steps": 10, # N: Manager acts every 10 steps
    "model_save_freq_steps": 100_000,
    "demo_steps_ranges": [ # Steps for expert demonstration
        (0, 10_000),
        (30_000, 40_000),
        (90_000, 100_000)
    ],
    "model_save_path": "./models_hrl/",
    "tensorboard_log": "./logs_hrl/"
}
