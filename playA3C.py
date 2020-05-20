import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from melee import Melee


options = dict(
    windows=True,
    render=True,
    speed=1,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

STATE_SIZE = 792
ACTION_SIZE = 30


def numpy_to_tensor(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Net, self).__init__()
        self.pi1 = nn.Linear(state_size, hidden_size)
        self.pi2 = nn.Linear(hidden_size, action_size)
        self.v1 = nn.Linear(state_size, hidden_size)
        self.v2 = nn.Linear(hidden_size, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, state):
        self.eval()
        logits, _ = self.forward(state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]


if __name__ == "__main__":
    policy_net = Net(STATE_SIZE, ACTION_SIZE)
    policy_net.load_state_dict(torch.load("checkpoints/agent.pth"))

    melee = Melee(**options)
    state = melee.reset()
    state_embed = np.array(melee.embed_state())

    while True:
        action = policy_net.choose_action(numpy_to_tensor(state_embed[None, :]))
        next_state = melee.step(action)