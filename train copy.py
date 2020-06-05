import time
import math
import random
import threading
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
#from replay_buffer import ReplayBuffer as ReplayBuffer
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

#learning_rate = 3e-5
learning_rate = 0.001
batch_size = 1024
save_every = 200
learn_every = 128

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500


def generate_frames(worker_id, network, replay_buffer, thread_dict):
    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    while True:
        try:
            actions = []
            for player_id in range(2):
                actions.append(network.act(states[player_id], thread_dict["epsilon"]))

            step_env_with_timeout = timeout(5)(lambda : env.step(actions))
            next_states, rewards, dones, _ = step_env_with_timeout()

            thread_dict["rewards"].append(rewards[0])

            for player_id in range(2):
                replay_buffer.add(states[player_id],
                                  actions[player_id],
                                  rewards[player_id],
                                  next_states[player_id],
                                  dones[player_id])
                thread_dict["frames_generated"] += 1
                if thread_dict["frames_generated"] % learn_every == 0:
                    thread_dict["learns_allowed"] += 1

            states = deepcopy(next_states)

        except KeyboardInterrupt:
            env.close()

        except:
            print("Dolphin crashed.")
            states = env.reset()


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")

    replay_buffer = ReplayBuffer(500000)

    thread_dict = {
        "frames_generated" : 0,
        "learns_allowed" : 0,
        "rewards" : [],
        "epsilon" : epsilon_start,
    }
    buffer_thread = threading.Thread(target=generate_frames, args=(0, network, replay_buffer, thread_dict))
    buffer_thread.start()

    learns = 0
    while True:
        if len(replay_buffer) > batch_size:
            while thread_dict["learns_allowed"] > 0:
                network.learn(replay_buffer)
                thread_dict["learns_allowed"] -= 1

                learns += 1
                if learns % save_every == 0:
                    print("Frames: {} / Learns: {} / Average Reward: {:.4f} / Epsilon: {:.2f}".format(
                        thread_dict["frames_generated"],
                        learns,
                        np.mean(thread_dict["rewards"]),
                        thread_dict["epsilon"],
                    ))
                    reward_buffer = []

                thread_dict["epsilon"] = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

        time.sleep(0.1)
