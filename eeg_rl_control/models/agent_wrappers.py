import numpy as np
import torch
import yaml
from types import SimpleNamespace
from eeg_rl_control.models.sac_core import SAC

class BaseAgent:
    def __init__(self, config_path, agent_name):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.agent_name = agent_name
        self.sac_params = config['sac_hyperparameters']
        self.action_map = config['action_maps'][f'{agent_name}_joints']
        self.joint_limits = config['joint_limits'][f'{agent_name}_joints']

        self.observation_dim = self._get_observation_dim()
        self.action_dim = len(self.action_map)

        action_space = SimpleNamespace(shape=[self.action_dim])

        self.sac_agent = SAC(self.observation_dim, action_space, self.sac_params['hidden_size'],
                             self.sac_params['learning_rate'], self.sac_params['gamma'],
                             self.sac_params['tau'], self.sac_params['alpha'])

    def _get_observation_dim(self):
        raise NotImplementedError

    def obs_to_tensor(self, obs):
        raise NotImplementedError

    def action_to_dict(self, action):
        scaled_action = self.scale_action(action)
        return {joint_name: scaled_action[i] for i, joint_name in enumerate(self.action_map)}

    def scale_action(self, action):
        """Scale action from [-1, 1] to [min, max]"""
        low = self.joint_limits['min']
        high = self.joint_limits['max']
        return low + (0.5 * (action + 1.0) * (high - low))

class WristShoulderAgent(BaseAgent):
    def __init__(self, config_path):
        super().__init__(config_path, 'wrist_shoulder')

    def _get_observation_dim(self):
        return 7  # imu_orientation_error_quat (4) + imu_linear_accel_error (3)

    def obs_to_tensor(self, obs):
        imu_quat = obs['imu_orientation_error_quat']
        imu_accel = obs['imu_linear_accel_error']
        return torch.FloatTensor(np.concatenate([imu_quat, imu_accel]))

class FingerAgent(BaseAgent):
    def __init__(self, config_path):
        super().__init__(config_path, 'fingers')

    def _get_observation_dim(self):
        return 4  # 4 finger flex errors

    def obs_to_tensor(self, obs):
        flex_errors = [obs[f'finger_{i}_flex_error'] for i in range(4)]
        return torch.FloatTensor(np.array(flex_errors))

class ThumbPalmAgent(BaseAgent):
    def __init__(self, config_path):
        super().__init__(config_path, 'thumb_palm')

    def _get_observation_dim(self):
        return 1  # thumb_flex_error

    def obs_to_tensor(self, obs):
        return torch.FloatTensor([obs['thumb_flex_error']])
