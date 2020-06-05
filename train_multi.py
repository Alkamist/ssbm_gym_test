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

learning_rate = 0.001
num_workers = 4
batch_size = 512
learn_every = 16
save_every = 200

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 1000


def generate_frames(worker_id, shared_state_dict, frame_queue, epsilon):
    env = MeleeEnv(worker_id=worker_id, **melee_options)

    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    policy_net.eval()

    states = env.reset()

    with torch.no_grad():
        while True:
            try:
                policy_net.load_state_dict(shared_state_dict.state_dict())

                actions = []
                for player_id in range(2):
                    if random.random() > epsilon.value:
                        state = torch.tensor(states[player_id], dtype=torch.float32, device=device).unsqueeze(0)
                        actions.append(policy_net(state).max(1)[1].item())
                    else:
                        actions.append(random.randrange(MeleeEnv.num_actions))

                step_env_with_timeout = timeout(5)(lambda : env.step(actions))
                next_states, rewards, dones, _ = step_env_with_timeout()

                for player_id in range(2):
                    frame_queue.put((states[player_id],
                                    actions[player_id],
                                    rewards[player_id],
                                    next_states[player_id],
                                    dones[player_id]))

                states = deepcopy(next_states)

            except KeyboardInterrupt:
                env.close()

            except:
                print("A dolphin worker crashed.")
                states = env.reset()


def add_frames_to_buffer(replay_buffer, thread_dict, frame_queue):
    frames_generated = 0
    while True:
        frame = frame_queue.get()
        replay_buffer.add(*frame)
        thread_dict["rewards"].append(frame[2])
        frames_generated += 1
        if frames_generated % learn_every == 0:
            thread_dict["learns_allowed"] += 1


if __name__ == "__main__":
    epsilon = mp.Value('d', epsilon_start)

    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    replay_buffer = ReplayBuffer(100000)
    frame_queue = mp.Queue()
    frame_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, shared_state_dict, frame_queue, epsilon))
        p.start()
        frame_processes.append(p)

    thread_dict = {
        "learns_allowed" : 0,
        "rewards" : [],
    }
    buffer_thread = threading.Thread(target=add_frames_to_buffer, args=(replay_buffer, thread_dict, frame_queue))
    buffer_thread.start()

    learns = 0
    while True:
        while thread_dict["learns_allowed"] > 0:
            network.learn(replay_buffer)
            shared_state_dict.load_state_dict(network.policy_net.state_dict())
            thread_dict["learns_allowed"] -= 1
            learns += 1

            epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

            if learns % save_every == 0:
                print("Frames: {} / Learns: {} / Average Reward: {:.4f} / Epsilon: {:.2f}".format(
                    learns * learn_every,
                    learns,
                    np.mean(thread_dict["rewards"]),
                    epsilon.value,
                ))
                thread_dict["rewards"] = []

        else:
            time.sleep(0.1)

    buffer_thread.join()
    for p in frame_processes:
        p.join()