import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#def initialize_weights(module):
#    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
#        torch.nn.init.kaiming_uniform_(module.weight)
#        if module.bias is not None:
#            torch.nn.init.constant_(module.bias, 0)


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Policy, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )#.apply(initialize_weights)

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

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            if random.random() > epsilon:
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.policy_net(state).max(1)[1].item()
            else:
                return random.randrange(self.action_size)

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

        batch = replay_buffer.sample(self.batch_size)

        #batch, indices, weights = replay_buffer.sample(self.batch_size)
        #weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = reward_batch + (next_state_values * self.gamma) * (1.0 - dones_batch)

        loss = self.loss_criterion(state_action_values, expected_state_action_values)

        #loss = self.loss_criterion(state_action_values, expected_state_action_values) * weights
        #priorities = loss
        #loss = loss.mean()
        #replay_buffer.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1