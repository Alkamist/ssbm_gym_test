import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, use_dueling_net):
        super(DQN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.use_dueling_net = use_dueling_net

        self.feature_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).apply(initialize_weights_he)

        if self.use_dueling_net:
            self.advantage_net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
            )
            self.baseline_net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
            )

    def forward(self, state):
        features = self.feature_net(state)

        if self.use_dueling_net:
            baselines = self.baseline_net(features)
            advantages = self.advantage_net(features)
            output = baselines + advantages - advantages.mean()
        else:
            output = self.net(features)

        return output


class DQNLearner():
    def __init__(self, state_size, action_size, hidden_size, batch_size, device, learning_rate,
                 gamma, grad_norm_clipping, target_update_frequency, use_dueling_net):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.device = device

        self.policy_net = DQN(
            input_size=self.state_size,
            output_size=self.action_size,
            hidden_size=self.hidden_size,
            use_dueling_net=use_dueling_net,
        ).to(self.device)
        self.policy_net.train()

        self.target_net = DQN(
            input_size=self.state_size,
            output_size=self.action_size,
            hidden_size=self.hidden_size,
            use_dueling_net=use_dueling_net,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
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
        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_state_values = self.target_net(next_states).max(1)[0].detach()

        expected_state_action_values = rewards.squeeze(1) + (next_state_values * self.gamma) * (1.0 - dones.squeeze(1))

        loss = self.loss_criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1