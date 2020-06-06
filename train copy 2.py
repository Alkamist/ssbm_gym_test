import math
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
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#learning_rate = 3e-5
learning_rate = 0.0001
batch_size = 64
learn_every = 2
print_every = 200

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 3000


def test_network(network, testing_thread_dict):
    env = MeleeEnv(worker_id=512, **melee_options)
    states = env.reset()

    while True:
        #actions = []
        #for player_id in range(2):
        #    actions.append(network.act(states[player_id], 0.0))

        actions = [network.act(states[0], 0.0), 0]

        next_states, rewards, _, _ = env.step(actions)

        testing_thread_dict["total_rewards"] += rewards[0]
        testing_thread_dict["frames_since_print"] += 1

        #for player_id in range(2):
        #    testing_thread_dict["total_rewards"] += rewards[player_id]
        #    testing_thread_dict["frames_since_print"] += 1

        states = deepcopy(next_states)


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")

    replay_buffer = ReplayBuffer(100000)

    testing_thread_dict = {
        "frames_since_print" : 0,
        "total_rewards" : 0.0,
    }
    testing_thread = threading.Thread(target=test_network, args=(network, testing_thread_dict))
    testing_thread.start()

    env = MeleeEnv(worker_id=0, **melee_options)
    states = env.reset()

    loops = 0
    learns = 0
    epsilon = epsilon_start
    while True:
        loops += 1

        #actions = []
        #for player_id in range(2):
        #    actions.append(network.act(states[player_id], epsilon))

        actions = [network.act(states[0], epsilon), 0]

        next_states, rewards, dones, _ = env.step(actions)

        replay_buffer.add(states[0],
                          actions[0],
                          rewards[0],
                          next_states[0],
                          dones[0])

#        for player_id in range(2):
#            replay_buffer.add(states[player_id],
#                              actions[player_id],
#                              rewards[player_id],
#                              next_states[player_id],
#                              dones[player_id])

        states = deepcopy(next_states)

        if loops % learn_every == 0:
            network.learn(replay_buffer)

            learns += 1
            if learns % print_every == 0:
                #network.save("checkpoints/agent" + str(learns) + ".pth")

                if testing_thread_dict["frames_since_print"] > 0:
                    average_reward = testing_thread_dict["total_rewards"] / testing_thread_dict["frames_since_print"]
                    testing_thread_dict["total_rewards"] = 0.0
                    testing_thread_dict["frames_since_print"] = 0
                else:
                    average_reward = 0.0

                print("Frames: {} / Learns: {} / Average Reward: {:.4f}".format(
                    loops * 2,
                    learns,
                    average_reward,
                ))

            #epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

    testing_thread.join()
