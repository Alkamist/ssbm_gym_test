import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import random


#class ResidualBlock(nn.Module):
#    """ https://arxiv.org/abs/1806.10909 """
#    def __init__(self, input_size, hidden_size):
#        super(ResidualBlock, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.mlp = nn.Sequential(
#            nn.Linear(self.input_size, self.hidden_size),
#            nn.ReLU(True),
#            nn.Linear(self.hidden_size, self.input_size),
#            nn.ReLU(True),
#        )
#
#    def forward(self, x):
#        return x + self.mlp(x)
#
#
#class Policy(nn.Module):
#    def __init__(self, input_size, output_size, hidden_size=64):
#        super(Policy, self).__init__()
#        self.res1 = ResidualBlock(input_size, hidden_size)
#        self.res2 = ResidualBlock(input_size, hidden_size)
#        self.output = nn.Linear(input_size, output_size)
#
#    def forward(self, x):
#        x = F.relu(self.res1(x))
#        x = F.relu(self.res2(x))
#        return self.output(x)


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN():
    def __init__(self, state_size, action_size, device, lr=3e-5, gamma=0.99, target_update_frequency=2500):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.policy_net = Policy(state_size, action_size).to(self.device)
        self.target_net = Policy(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
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

    def learn(self, batch):
        self.policy_net.train()
        self.target_net.eval()

        state_action_values = self.policy_net(batch.state).gather(1, batch.action)
        next_state_values = self.target_net(batch.next_state).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + batch.reward

        loss = self.loss_criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1
