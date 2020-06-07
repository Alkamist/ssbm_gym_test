import math
import time
import random
import threading
from copy import deepcopy

import torch

from melee_env import MeleeEnv
#from replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
from replay_buffer import ReplayBuffer as ReplayBuffer
from DQN import DQN


melee_options = dict(
    render=True,
    speed=0,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#learning_rate = 3e-5
learning_rate = 0.0001
batch_size = 16
print_every = 2000


def generate_frames(worker_id, network, replay_buffer, thread_dict, epsilon):
    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    while True:
        #actions = []
        #for player_id in range(2):
        #    actions.append(network.act(states[player_id], 0.0))

        actions = [network.act(states[0], epsilon), 0]

        next_states, rewards, dones, _ = env.step(actions)

        replay_buffer.add(states[0],
                          actions[0],
                          rewards[0],
                          next_states[0],
                          dones[0])

        thread_dict["total_rewards"] += rewards[0]
        thread_dict["frames_generated"] += 1
        thread_dict["learns_allowed"] += 1

        #for player_id in range(2):
        #    testing_thread_dict["total_rewards"] += rewards[player_id]
        #    testing_thread_dict["frames_since_print"] += 1

        states = deepcopy(next_states)


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")


    random_replay_buffer = ReplayBuffer(250000)
    random_thread_dict = {
        "total_rewards" : 0.0,
        "frames_generated" : 0,
        "learns_allowed" : 0,
    }
    random_thread = threading.Thread(target=generate_frames, args=(512, network, random_replay_buffer, random_thread_dict, 1.0))
    random_thread.start()


    policy_replay_buffer = ReplayBuffer(3600)
    policy_thread_dict = {
        "total_rewards" : 0.0,
        "frames_generated" : 0,
        "learns_allowed" : 0,
    }
    policy_thread = threading.Thread(target=generate_frames, args=(0, network, policy_replay_buffer, policy_thread_dict, 0.0))
    policy_thread.start()


    loops = 0
    random_learns = 0
    policy_learns = 0
    policy_frames_generated = 0
    while True:
        while random_thread_dict["learns_allowed"] > 0:
            network.learn(random_replay_buffer)
            random_thread_dict["learns_allowed"] -= 1
            random_learns += 1

            if policy_thread_dict["learns_allowed"] > 0:
                network.learn(policy_thread_dict)
                policy_thread_dict["learns_allowed"] -= 1
                policy_learns += 1

            loops += 1
            if loops % print_every == 0:
                network.save("checkpoints/agent" + str(random_learns + policy_learns) + ".pth")

                if policy_thread_dict["frames_generated"] > 0:
                    policy_frames_generated += policy_thread_dict["frames_generated"]
                    average_reward = policy_thread_dict["total_rewards"] / policy_thread_dict["frames_generated"]
                    policy_thread_dict["total_rewards"] = 0.0
                    policy_thread_dict["frames_generated"] = 0
                else:
                    average_reward = 0.0

                print("Frames: {} / Learns: {} / Average Reward: {:.4f}".format(
                    random_thread_dict["frames_generated"] + policy_frames_generated,
                    random_learns + policy_learns,
                    average_reward,
                ))

        time.sleep(0.1)


    random_thread.join()
    policy_thread.join()
