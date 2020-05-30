from collections import namedtuple

import torch
import torch.nn as nn


NetworkOutput = namedtuple("NetworkOutput", ("value", "reward", "policy_logits", "hidden_state"))


class MuZeroNet(nn.Module):
    def __init__(self, input_size, num_actions):
        super(MuZeroNet, self).__init__()
        self.hx_size = 32
        self._representation = nn.Sequential(nn.Linear(input_size, self.hx_size),
                                             nn.Tanh())
        self._dynamics_state = nn.Sequential(nn.Linear(self.hx_size + num_actions, 64),
                                             nn.Tanh(),
                                             nn.Linear(64, self.hx_size),
                                             nn.Tanh())
        self._dynamics_reward = nn.Sequential(nn.Linear(self.hx_size + num_actions, 64),
                                              nn.LeakyReLU(),
                                              nn.Linear(64, 1))
        self._prediction_actor = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, num_actions))
        self._prediction_value = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, 1))
        self.num_actions = num_actions

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.num_actions), dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward

    def initial_inference(self, obs):
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

        return NetworkOutput(value, 0, actor_logit, state)

    def recurrent_inference(self, hidden_state, action):
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)

        return NetworkOutput(value, reward, actor_logit, state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
