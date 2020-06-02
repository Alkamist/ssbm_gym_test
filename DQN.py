import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """ https://arxiv.org/abs/1806.10909 """
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.input_size),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.mlp(x)


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(Policy, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.advantage = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state):
        features = self.features(state)
        values = self.value(features)
        advantages = self.advantage(features)
        return values + (advantages - advantages.mean())


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
        self.num_ai_players = 2

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            if random.random() > epsilon:
                return self.policy_net(state).max(1)[1]
            else:
                return torch.tensor([random.randrange(self.action_size) for _ in range(self.num_ai_players)], device=self.device, dtype=torch.long)

    def save(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def evaluate(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def learn(self, replay_buffer):
        if len(replay_buffer) <= self.batch_size:
            return

        self.policy_net.train()
        self.target_net.eval()

        batch, indices, weights = replay_buffer.sample(self.batch_size)
        #batch = replay_buffer.sample(self.batch_size)

        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = ((next_state_values * self.gamma) + reward_batch)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        loss = (self.loss_criterion(state_action_values, expected_state_action_values) * weights).mean()

        #loss = self.loss_criterion(state_action_values, expected_state_action_values)

        errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
        replay_buffer.update_priorities(indices, errors)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1
