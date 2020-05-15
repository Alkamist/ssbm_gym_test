import math
import random
import numpy as np

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BitFlipEnv():
    def __init__(self, n = 8):
        self.n = n
        self.init_state = torch.randint(2, size=(n,))
        self.target_state = torch.randint(2, size=(n,))
        while np.array_equal(self.init_state, self.target_state):
            self.target_state = torch.randint(2, size=(n,))
        self.curr_state = self.init_state.clone()

    def step(self, action):
        self.curr_state[action] = 1 - self.curr_state[action]
        if np.array_equal(self.curr_state, self.target_state):
            return self.curr_state.clone(), 0
        else:
            return self.curr_state.clone(), -1

    def reset(self):
        self.init_state = torch.randint(2, size=(self.n,))
        self.target_state = torch.randint(2, size=(self.n,))
        while np.array_equal(self.init_state, self.target_state):
            self.target_state = torch.randint(2, size=(self.n,))
        self.curr_state = self.init_state.clone()

env = BitFlipEnv(n=10)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'goal'))
HindsightTransition = namedtuple('HindsightTransition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity = 1e5):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

NUM_BITS = 8
HIDDEN_SIZE = 256

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.ln1 = nn.Linear(NUM_BITS*2, HIDDEN_SIZE)
        self.ln2 = nn.Linear(HIDDEN_SIZE, NUM_BITS)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return x

BATCH_SIZE = 128 # batch size for training
GAMMA = 0.999 # discount factor
EPS_START = 0.95 # eps greedy parameter
EPS_END = 0.05
TARGET_UPDATE = 50 # number of epochs before target network weights are updated to policy network weights
steps_done = 0 # for decayin eps

policy_net = FNN().to(device)
target_net = FNN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1e6)

def select_action(state, goal, greedy=False):
    global steps_done
    sample = random.random()
    state_goal = torch.cat((state, goal))

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if greedy:
        with torch.no_grad():
            return policy_net(state_goal).argmax().view(1,1)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state_goal).argmax().view(1,1)
    else:
        return torch.tensor([[random.randrange(NUM_BITS)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                           batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                      if s is not None])

    # extract state, action, reward, goal from randomly sampled transitions
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    goal_batch = torch.stack(batch.goal)

    # concatenate state and goal for network input
    state_goal_batch = torch.cat((state_batch, goal_batch), 1)
    non_final_next_states_goal = torch.cat((non_final_next_states, goal_batch), 1)

    # get current state action values
    state_action_values = policy_net(state_goal_batch).gather(1, action_batch)

    # get next state values according to target_network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states_goal).max(1)[0].detach()

    # calculate expected q value of current state acc to target_network
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # find huber loss using curr q-value and expected q-value
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

env = BitFlipEnv(NUM_BITS)
success = 0

for i_episode in range(num_episodes):
    env.reset()
    state = env.init_state
    goal = env.target_state
    transitions = []
    episode_success = False

    for t in range(NUM_BITS):

        if episode_success:
            continue

        action = select_action(state, goal)
        next_state, reward = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # add transition to replay memory
        memory.push(state, action, next_state, reward, goal)

        # store transition without goal state for hindsight
        transitions.append(HindsightTransition(state, action, next_state, reward))

        state = next_state

        optimize_model()
        if reward == 0:
            if episode_success:
                continue
            else:
                episode_success = True
                success += 1

    # add hindsight transitions to the replay memory
    if not episode_success:
        # failed episode store the last visited state as new goal
        new_goal_state = state.clone()
        if not np.array_equal(new_goal_state, goal):
            for i in range(NUM_BITS):
                # if goal state achieved
                if np.array_equal(transitions[i].next_state, new_goal_state):
                    memory.push(transitions[i].state, transitions[i].action, transitions[i].next_state, torch.tensor([0]), new_goal_state)
                    optimize_model()
                    break

                memory.push(transitions[i].state, transitions[i].action, transitions[i].next_state, transitions[i].reward, new_goal_state)
                optimize_model()

    # update the target networks weights
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())