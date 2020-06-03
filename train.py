import time
import math
import queue
import random
import threading
from copy import deepcopy
#from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
from DQN import DQN, Policy
from timeout import timeout


melee_options = dict(
    render=True,
    speed=0,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learn_every = 8
save_every = 200
replay_buffer_size = 250000

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 1200


if __name__ == "__main__":
    reward_buffer = []

    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=0.0001)
    #network.load("checkpoints/agent.pth")

    replay_buffer = ReplayBuffer(replay_buffer_size)

    env = MeleeEnv(**melee_options)
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32, device=device)

    epsilon = epsilon_start
    learn_iterations = 0
    while True:
        try:
            with torch.no_grad():
                for _ in range(learn_every):
                    actions = network.act(states, epsilon=epsilon)
                    step_env_with_timeout = timeout(5)(lambda : env.step(actions))
                    next_states, rewards, dones, _ = step_env_with_timeout()

                    for player_id in range(2):
                        replay_buffer.add(states[player_id].cpu().numpy(),
                                          actions[player_id].item(),
                                          rewards[player_id],
                                          next_states[player_id],
                                          dones[player_id])

                    reward_buffer.append(rewards[0])

                    states = deepcopy(next_states)
                    states = torch.tensor(states, dtype=torch.float32, device=device)

            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learn_iterations / epsilon_decay)

            learn_iterations += 1
            network.learn(replay_buffer)

            if learn_iterations % save_every == 0:
                #network.save("checkpoints/agent" + str(learn_iterations) + ".pth")
                print("Total Frames: {} / Learn Iterations: {} / Average Reward: {:.4f} / Epsilon: {:.2f}".format(
                    learn_iterations * learn_every * 2,
                    learn_iterations,
                    np.mean(reward_buffer),
                    epsilon
                ))
                reward_buffer = []

        except KeyboardInterrupt:
            env.close()

        except:
            states = env.reset()
            states = torch.tensor(states, dtype=torch.float32, device=device)