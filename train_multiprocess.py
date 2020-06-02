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
#from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
from DQN import DQN, Policy
from timeout import timeout


melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 3
batch_size = 64
learn_every = 8
save_every = 8000
replay_buffer_size = 250000

epsilon_start = 1.0
epsilon_end = 1.0
epsilon_decay = 10000


def test_policy(shared_state_dict, info_queue):
    reward_buffer = []

    policy = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device)

    testing_options = deepcopy(melee_options)
    testing_options["render"] = True
    env = MeleeEnv(worker_id=1024, **testing_options)
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32, device=device)

    with torch.no_grad():
        while True:
            try:
                try:
                    test_signal = info_queue.get(block=False)
                    if test_signal:
                        print("Average Reward: %.4f" % np.mean(reward_buffer))
                        policy.load_state_dict(shared_state_dict.state_dict())
                        reward_buffer = []
                except queue.Empty:
                    test_signal = False

                actions = policy(states).max(1)[1]
                step_env_with_timeout = timeout(5)(lambda : env.step(actions))
                states, rewards, _, _ = step_env_with_timeout()
                states = torch.tensor(states, dtype=torch.float32, device=device)

                reward_buffer.append(rewards[0])

            except KeyboardInterrupt:
                env.close()

            except:
                print("The testing dolphin instance crashed.")
                states = env.reset()
                states = torch.tensor(states, dtype=torch.float32, device=device)
                return


def generate_frames(worker_id, frame_queue):
    env = MeleeEnv(worker_id=worker_id, **melee_options)

    states = env.reset()

    with torch.no_grad():
        while True:
            try:
                actions = [random.randrange(MeleeEnv.num_actions), random.randrange(MeleeEnv.num_actions)]

                step_env_with_timeout = timeout(5)(lambda : env.step(actions))
                next_states, rewards, _, _ = step_env_with_timeout()

                for player_id in range(2):
                    frame_queue.put((states[player_id],
                                    actions[player_id],
                                    next_states[player_id],
                                    rewards[player_id]))

                states = deepcopy(next_states)

            except KeyboardInterrupt:
                env.close()

            except:
                env.close()
                frame_queue.put(worker_id)
                return


def add_frames_to_replay_buffer(info_flags, replay_buffer, frame_queue):
    with torch.no_grad():
        while True:
            frame = frame_queue.get()

            if isinstance(frame, tuple):
                state, action, next_state, reward = frame
                replay_buffer.add(state, action, next_state, reward)
            else:
                info_flags["dolphin_crashed"] = frame

            info_flags["frames_generated"] += 1

            if info_flags["frames_generated"] % learn_every == 0:
                info_flags["learns_allowed"] += 1


def create_worker(worker_id, frame_queue):
    p = mp.Process(target=generate_frames, args=(worker_id, frame_queue))
    p.start()
    return p


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, device, lr=0.0001)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)

    frame_queue = mp.Queue()

    generator_processes = []
    for worker_id in range(num_workers):
        generator_processes.append(create_worker(worker_id, frame_queue))

    # Start transferring frames to the replay buffer on a separate thread
    # so it doesn't slow down learning.
    info_flags = {
        "frames_generated" : 0,
        "learns_allowed" : 0,
        "dolphin_crashed" : None,
    }
    replay_buffer_frame_thread = threading.Thread(
        target=add_frames_to_replay_buffer,
        args=(info_flags, replay_buffer, frame_queue)
    )
    replay_buffer_frame_thread.start()

    # Start the testing process asynchronously so it doesn't slow down learning.
    testing_queue = mp.Queue()
    testing_process = mp.Process(
        target=test_policy,
        args=(shared_state_dict, testing_queue)
    )
    testing_process.start()

    need_to_save = False
    learn_iterations = 0
    while True:
        # Reset any crashed dolphin processes.
        if info_flags["dolphin_crashed"] is not None:
            print("A dolphin instance crashed.")
            generator_processes[info_flags["dolphin_crashed"]].terminate()
            generator_processes[info_flags["dolphin_crashed"]] = create_worker(info_flags["dolphin_crashed"], frame_queue)
            info_flags["dolphin_crashed"] = None

        # Learn for one iteration.
        if info_flags["learns_allowed"] > 0 and len(replay_buffer) > batch_size:
            network.learn(replay_buffer)
            shared_state_dict.load_state_dict(network.policy_net.state_dict())

            learn_iterations += 1
            info_flags["learns_allowed"] -= 1
            need_to_save = True

        if learn_iterations % 400 == 0 and learn_iterations > 0 and need_to_save:
            testing_queue.put(True)

        # Save every X learn iterations.
        if learn_iterations % save_every == 0 and learn_iterations > 0 and need_to_save:
            #network.save("checkpoints/agent" + str(learn_iterations) + ".pth")
            need_to_save = False
            print("Total Frames: {} / Learn Iterations: {}".format(
                info_flags["frames_generated"],
                learn_iterations,
            ))

    testing_process.join()
    replay_buffer_frame_thread.join()
    for p in generator_processes:
        p.join()