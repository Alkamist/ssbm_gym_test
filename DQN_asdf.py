import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(Policy, self).__init__()

        feature_size = 256
        hidden_size = 128

        self.features = nn.Sequential(
            nn.Linear(input_size, feature_size),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.to(device)

    def forward(self, state):
        features = self.features(state)
        values = self.value(features)
        advantages = self.advantage(features)
        return values + advantages - advantages.mean()


class DQN():
    def __init__(self, state_size, action_size, batch_size, device, lr=3e-5, gamma=0.997, target_update_frequency=2500):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.policy_net = Policy(state_size, action_size, device=self.device)
        self.target_net = Policy(state_size, action_size, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_criterion = torch.nn.SmoothL1Loss()
        self.target_update_frequency = target_update_frequency
        self.learn_iterations = 0

    def save(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def evaluate(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def learn(self, states, actions, rewards, next_states, dones):
        self.policy_net.train()
        self.target_net.eval()

        state_action_values = self.policy_net(states).gather(2, actions).squeeze(2)
        next_state_values = self.target_net(next_states).max(2)[0].detach()

        expected_state_action_values = rewards + (next_state_values * self.gamma) * (1.0 - dones)

        loss = self.loss_criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1