import math
import random
from collections import deque

import torch
import numpy as np

from melee import Melee
from replay_buffer import ReplayBuffer
from DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

options = dict(
    windows=True,
    render=False,
    speed=0,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

training_cycles = 300
memories_per_cycle = 30000
frames_to_check_performance = 10800

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 100

batch_size = 32
state_size = 792
action_size = 30

goal = [25.0, 0.0]
goal_embed = [goal[0] / 100.0, goal[1] / 100.0]

def calculate_reward(state_embed, goal_embed):
    def calculate_distance(x0, y0, x1, y1):
        return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    max_distance_for_reward = 5.0
    x0 = state_embed[0]
    y0 = state_embed[1]
    x1 = goal_embed[0]
    y1 = goal_embed[1]
    reward = 1.0 if calculate_distance(x0, y0, x1, y1) < (max_distance_for_reward / 100.0) else -1.0
    return reward

def perform_action(melee, dqn, epsilon):
    state_embed = torch.as_tensor([melee.embed_state()], device=device, dtype=torch.float32)
    action = dqn.act(state_embed, epsilon=epsilon)
    next_state = melee.step(action)
    next_state_embed = torch.as_tensor([melee.embed_state()], device=device, dtype=torch.float32)
    reward = torch.as_tensor([[calculate_reward(state_embed[0], goal_embed)]], device=device, dtype=torch.float32)
    return state_embed, action, next_state_embed, reward, next_state

def create_memories(dqn, epsilon, steps):
    melee = Melee(**options)
    melee.reset()
    for _ in range(steps):
        state_embed, action, next_state_embed, reward, _ = perform_action(melee, dqn, epsilon)
        replay_buffer.add(state_embed, action, next_state_embed, reward)
    melee.close()

def train_network(dqn, replay_buffer, steps):
    if len(replay_buffer) > batch_size:
        for _ in range(steps):
            dqn.learn(replay_buffer.sample(batch_size=batch_size), batch_size)

def check_network_performance(dqn, steps):
    melee = Melee(**options)
    state = melee.reset()
    rewards = deque(maxlen=steps)
    for _ in range(steps):
        _, _, _, reward, next_state = perform_action(melee, dqn, 0.0)
        state = next_state
        rewards.append(reward[0].item())
    print("R: %.4f" % np.mean(rewards), "X: %.2f" % state.players[0].x)
    melee.close()

if __name__ == "__main__":
    replay_buffer = ReplayBuffer(buffer_size=100000)
    #replay_buffer.load("replay_memories/memories")
    #print("Loaded memory.")

    dqn = DQN(state_size=state_size, action_size=action_size)
    #dqn.load("checkpoints/agent.pth")

    for cycle_count in range(training_cycles):
        #epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * cycle_count / epsilon_decay)
        #print("Epsilon: %.2f" % epsilon)
        epsilon = 1.0

        create_memories(dqn, epsilon, steps=memories_per_cycle)
        #print("Memories Created")

        train_network(dqn, replay_buffer, steps=3*int(memories_per_cycle/batch_size))
        #if cycle_count > 0 and cycle_count % 4 == 0:
        dqn.save("checkpoints/" + str(cycle_count) + ".pth")

        check_network_performance(dqn, steps=frames_to_check_performance)