import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from eeg_rl_control.models.networks import GaussianPolicy, QNetwork
import numpy as np


class ReplayBuffer:
    def __init__(self, size, input_shape, n_actions):
        self.size = size
        self.counter = 0
        self.state_memory = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.next_state_memory = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.counter % self.size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.terminal_memory[idx] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, next_states, dones


class SAC(object):
    def __init__(self, num_inputs, action_space, hidden_size, lr, gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size)
        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size)

        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        mask_batch = torch.FloatTensor(np.float32(mask_batch)).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Soft update for target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))
        self.critic_target.load_state_dict(self.critic.state_dict())
