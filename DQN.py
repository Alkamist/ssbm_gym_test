import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#class Policy(nn.Module):
#    def __init__(self, input_size, output_size, hidden_size=512):
#        super(Policy, self).__init__()
#
#        self.features = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(),
#        )
#
#        self.value = nn.Sequential(
#            nn.Linear(hidden_size, hidden_size),
#            nn.ReLU(),
#            nn.Linear(hidden_size, 1),
#        )
#
#        self.advantage = nn.Sequential(
#            nn.Linear(hidden_size, hidden_size),
#            nn.ReLU(),
#            nn.Linear(hidden_size, output_size),
#        )
#
#    def forward(self, state):
#        features = self.features(state)
#        values = self.value(features)
#        advantages = self.advantage(features)
#        return values + advantages - advantages.mean()


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Policy, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, 1)
        self.lstm_state = None

        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def reset_rnn(self):
        self.lstm_state = None

    def forward(self, state):
        features = self.features(state)
        lstm_output, self.lstm_state = self.lstm(features, self.lstm_state)
        values = self.value(lstm_output)
        advantages = self.advantage(lstm_output)
        return values + advantages - advantages.mean()


class DQN():
    def __init__(self, state_size, action_size, batch_size, device, lr=3e-5, gamma=0.997, target_update_frequency=2500):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.policy_net = Policy(state_size, action_size).to(self.device)
        self.target_net = Policy(state_size, action_size).to(self.device)
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

    def reset_rnn(self):
        self.policy_net.reset_rnn()
        self.target_net.reset_rnn()

    def learn(self, states, actions, rewards, next_states, dones):
        self.policy_net.train()
        self.target_net.eval()

        state_batch = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(2)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(2, action_batch).squeeze(2)
        next_state_values = self.target_net(next_state_batch).max(2)[0].detach()

        expected_state_action_values = reward_batch + (next_state_values * self.gamma) * (1.0 - dones_batch)

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