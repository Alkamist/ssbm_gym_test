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
    def __init__(self, input_size, output_size, num_atoms, hidden_size=512):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.num_atoms = num_atoms

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #ResidualBlock(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_atoms),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #ResidualBlock(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_atoms * output_size),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        batch_size = state.size(0)
        features = self.features(state)
        values = self.value(features).view(batch_size, 1, self.num_atoms)
        advantages = self.advantage(features).view(batch_size, self.output_size, self.num_atoms)
        output = values + (advantages - advantages.mean(1, keepdim=True))
        return self.softmax(output.view(-1, self.num_atoms)).view(-1, self.output_size, self.num_atoms)


class DQN():
    def __init__(self, state_size, action_size, batch_size, device, lr=3e-5, gamma=0.997, target_update_frequency=2500):
        self.num_atoms = 51
        self.Vmin = -10
        self.Vmax = 10
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.policy_net = Policy(state_size, action_size, self.num_atoms).to(self.device)
        self.target_net = Policy(state_size, action_size, self.num_atoms).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_update_frequency = target_update_frequency
        self.learn_iterations = 0

    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            probabilities = self.policy_net(state)
            expected_value = probabilities * torch.linspace(self.Vmin, self.Vmax, self.num_atoms, device=self.device)
            return expected_value.sum(2).max(1)[1].item()
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

    def _projection_distribution(self, next_states, rewards, dones):
        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms, device=self.device)

        next_distribution = self.target_net(next_states) * support
        next_actions = next_distribution.sum(2).max(1)[1]
        next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(next_distribution.size(0), 1, next_distribution.size(2))
        next_distribution = next_distribution.gather(1, next_actions).squeeze(1)

        support = support.unsqueeze(0).expand_as(next_distribution).to(self.device)
        rewards = rewards.expand_as(next_distribution)
        dones = dones.expand_as(next_distribution)

        Tz = rewards + (1 - dones) * self.gamma * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size, device=self.device).long().unsqueeze(1).expand(self.batch_size, self.num_atoms)

        projection_distribution = torch.zeros(next_distribution.size(), device=self.device)
        projection_distribution.view(-1).index_add_(0, (l + offset).view(-1), (next_distribution * (u.float() - b)).view(-1))
        projection_distribution.view(-1).index_add_(0, (u + offset).view(-1), (next_distribution * (b - l.float())).view(-1))

        return projection_distribution

    def learn(self, replay_buffer):
        if len(replay_buffer) <= self.batch_size:
            return

        self.policy_net.train()
        self.target_net.eval()

        batch = replay_buffer.sample(self.batch_size)

        #batch, indices, weights = replay_buffer.sample(self.batch_size)
        #weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        projection_distribution = self._projection_distribution(next_state_batch, reward_batch, dones_batch)

        probability_distribution = self.policy_net(state_batch)
        actions = action_batch.unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        state_action_probabilities = probability_distribution.gather(1, actions).squeeze(1)
        state_action_probabilities.clamp_(0.01, 0.99)

        #loss_priorities = -((state_action_probabilities.log() * projection_distribution.detach()).sum(dim=1).unsqueeze(1) * weights)
        #loss = loss_priorities.mean()

        loss = -(state_action_probabilities.log() * projection_distribution.detach()).sum(dim=1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #replay_buffer.update_priorities(indices, torch.abs(loss_priorities.squeeze(1)).detach().cpu().numpy())

        self.learn_iterations += 1
