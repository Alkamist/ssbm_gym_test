import numpy as np

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


def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None].expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(taus[..., None] - (td_errors.detach() < 0).float()) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, output_size, num_cosines):
        super(CosineEmbeddingNetwork, self).__init__()

        self.output_size = output_size
        self.num_cosines = num_cosines

        self.net = nn.Sequential(
            nn.Linear(self.num_cosines, self.output_size),
            nn.ReLU()
        )

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(start=1, end=self.num_cosines+1, dtype=taus.dtype, device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, self.output_size)

        return tau_embeddings


class QuantileNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, use_dueling_net):
        super(QuantileNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.use_dueling_net = use_dueling_net

        if self.use_dueling_net:
            self.advantage_net = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
            )
            self.baseline_net = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
            )

    def forward(self, state_embeddings, tau_embeddings):
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        assert state_embeddings.shape[1] == tau_embeddings.shape[2]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.
        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, input_size).
        state_embeddings = state_embeddings.view(batch_size, 1, self.input_size)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(batch_size * N, self.input_size)

        # Calculate quantile values.
        if self.use_dueling_net:
            advantages = self.advantage_net(embeddings)
            baselines = self.baseline_net(embeddings)
            quantiles = baselines + advantages - advantages.mean(1, keepdim=True)
        else:
            quantiles = self.net(embeddings)

        return quantiles.view(batch_size, N, self.output_size)


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, use_dueling_net):
        super(DQN, self).__init__()

        self.K = 32
        self.num_cosines = 64

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.feature_net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).apply(initialize_weights_he)

        self.cosine_net = CosineEmbeddingNetwork(
            output_size=self.hidden_size,
            num_cosines=self.num_cosines,
        )

        self.quantile_net = QuantileNetwork(
            input_size=self.hidden_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            use_dueling_net=use_dueling_net,
        )

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.feature_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.feature_net(states)

        # Sample fractions.
        taus = torch.rand(batch_size, self.K, dtype=state_embeddings.dtype, device=state_embeddings.device)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.output_size)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.output_size)

        return q


class DQNLearner():
    def __init__(self, state_size, action_size, hidden_size, batch_size, device, learning_rate,
                 gamma, grad_norm_clipping, target_update_frequency, use_dueling_net):
        self.N = 64
        self.N_dash = 64
        self.kappa = 1.0

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.device = device

        self.policy_net = DQN(
            input_size=state_size,
            output_size=action_size,
            hidden_size=self.hidden_size,
            use_dueling_net=use_dueling_net,
        ).to(self.device)
        self.policy_net.train()

        self.target_net = DQN(
            input_size=state_size,
            output_size=action_size,
            hidden_size=self.hidden_size,
            use_dueling_net=use_dueling_net,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.target_update_frequency = target_update_frequency
        self.learn_iterations = 0

    def save(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def learn(self, states, actions, rewards, next_states, dones):
        state_embeddings = self.policy_net.feature_net(states)
        weights = None

        loss, _, _ = self._calculate_loss(state_embeddings, actions, rewards, next_states, dones, weights)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1

    def _calculate_loss(self, state_embeddings, actions, rewards, next_states, dones, weights):
        # Sample fractions.
        taus = torch.rand(self.batch_size, self.N, dtype=state_embeddings.dtype, device=state_embeddings.device)

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = evaluate_quantile_at_action(self.policy_net.calculate_quantiles(taus, state_embeddings=state_embeddings), actions)
        assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate Q values of next states.
            next_q = self.policy_net(states=next_states)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate features of next states.
            next_state_embeddings = self.target_net.feature_net(next_states)

            # Sample next fractions.
            tau_dashes = torch.rand(self.batch_size, self.N_dash, dtype=state_embeddings.dtype, device=state_embeddings.device)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(self.target_net.calculate_quantiles(tau_dashes, state_embeddings=next_state_embeddings), next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (1.0 - dones[..., None]) * self.gamma * next_sa_quantiles
            assert target_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

        quantile_huber_loss = calculate_quantile_huber_loss(td_errors, taus, weights, self.kappa)

        return quantile_huber_loss, next_q.detach().mean().item(), td_errors.detach().abs()
