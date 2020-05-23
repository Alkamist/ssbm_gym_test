import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64, rnn_hidden_size=256, rnn_layers=1):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        self.core = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(True),
            ResidualBlock(hidden_size, 32),
        )
        self.rnn = nn.GRU(action_size + hidden_size, rnn_hidden_size, rnn_layers)

        self.policy = nn.Linear(rnn_hidden_size, action_size)
        self.value = nn.Linear(rnn_hidden_size, 1)

        self.rnn_hidden = None
        self.prev_action = None

    def _forward(self, observation, rnn_hidden):
        x = self.core(observation)
        if self.prev_action is None:
            action = torch.zeros(1, x.shape[1], self.action_size).to(x.device)
        else:
            action = torch.nn.functional.one_hot(self.prev_action, self.action_size).to(x.device).float()
        action = action.repeat((int(x.shape[0] / action.shape[0]), 1, 1))
        y = torch.cat([action, x], dim=-1)

        h, rnn_hidden = self.rnn(y, rnn_hidden)
        logits = self.policy(h[-1:])
        dist = Categorical(logits=logits)
        return dist, rnn_hidden

    def forward(self, observation):
        dist, self.rnn_hidden = self._forward(observation, self.rnn_hidden)
        action = dist.sample()
        self.prev_action = action
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def act(self, observation):
        dist, self.rnn_hidden = self._forward(observation, self.rnn_hidden)
        action = dist.sample()
        self.prev_action = action
        return action.detach()

    def evaluate_actions(self, observations, actions, rnn_hidden=None):
        n = int((observations.shape[0] - 1) / (actions.shape[0] - 1))
        x = self.core(observations)
        prev_actions = self.get_prev_actions(actions, n).squeeze(-1)
        y = torch.cat([prev_actions, x], dim=-1)
        h, _ = self.rnn(y, rnn_hidden)
        h = h[list(range(0, observations.shape[0], n))]
        values = self.value(h).squeeze(-1)
        logits = self.policy(h[:-1])
        dist = Categorical(logits=logits)
        actions_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, actions_log_probs, entropy

    def get_prev_actions(self, actions, n):
        first_prev_action = torch.zeros((1, actions.shape[1], self.action_size), device=actions.device)
        next_prev_actions = torch.nn.functional.one_hot(actions.repeat_interleave(n, dim=0), self.action_size).float()
        prev_actions = torch.cat([first_prev_action, next_prev_actions], dim=0)
        return prev_actions

    def reset_rnn(self):
        self.rnn_hidden = None
        self.prev_action = None

class ResidualBlock(nn.Module):
    """
    https://arxiv.org/abs/1806.10909
    """
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, data_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x + self.mlp(x)

def partial_load(model, path, debug=True):
    old_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    if debug:
        print("Non-matching keys: ", {k for k, _ in old_dict.items() if not (k in model_dict and model_dict[k].shape == old_dict[k].shape)})
    old_dict = {k: v for k, v in old_dict.items() if k in model_dict and model_dict[k].shape == old_dict[k].shape}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict)