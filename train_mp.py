import time
import math
import random
import threading
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from storage_buffer import StorageBuffer
from DQN import DQN, Policy


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

num_workers = 1
batch_size = 16
memory_size = 100000
save_every = 2000

gamma = 0.9995
learning_rate = 0.0001

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 5000


def generate_frames(worker_id, shared_state_dict, frame_queue, epsilon):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions, device=device)
    policy_net.eval()

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    with torch.no_grad():
        while True:
            policy_net.load_state_dict(shared_state_dict.state_dict())

            if random.random() <= epsilon.value:
                action = random.randrange(MeleeEnv.num_actions)
            else:
                state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
                action = policy_net(state).max(2)[1].item()

            actions = [action, 0]
            next_states, rewards, dones, _ = env.step(actions)

            frame_queue.put((states[0],
                             action,
                             rewards[0],
                             next_states[0],
                             dones[0]))

            states = deepcopy(next_states)


def prepare_batches(storage_buffer, thread_dict, frame_queue):
    while True:
        frame = frame_queue.get()
        storage_buffer.add_item((torch.tensor([[frame[0]]], dtype=torch.float32, device=device),
                                 torch.tensor([[[frame[1]]]], dtype=torch.long, device=device),
                                 torch.tensor([[frame[2]]], dtype=torch.float32, device=device),
                                 torch.tensor([[frame[3]]], dtype=torch.float32, device=device),
                                 torch.tensor([[frame[4]]], dtype=torch.float32, device=device)))

        thread_dict["frames_generated"] += 1
        thread_dict["rewards"].append(frame[2])

        if len(storage_buffer) > batch_size:
            batch_of_frames = storage_buffer.sample_batch(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch_of_frames)

            thread_dict["batches"].append((
                torch.cat(state_batch, dim=1),
                torch.cat(action_batch, dim=1),
                torch.cat(reward_batch, dim=1),
                torch.cat(next_state_batch, dim=1),
                torch.cat(done_batch, dim=1),
            ))


if __name__ == "__main__":
    learner = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate, gamma=gamma)
    #learner.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions, device="cpu")
    shared_state_dict.load_state_dict(learner.policy_net.state_dict())
    shared_state_dict.share_memory()

    storage_buffer = StorageBuffer(memory_size)
    frame_queue = mp.Queue(maxsize=1)
    epsilon = mp.Value("d", epsilon_start)

    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, shared_state_dict, frame_queue, epsilon))
        p.start()
        generator_processes.append(p)

    thread_dict = {
        "batches" : deque(maxlen=8),
        "frames_generated" : 0,
        "rewards" : deque(maxlen=3600),
    }
    batch_thread = threading.Thread(target=prepare_batches, args=(storage_buffer, thread_dict, frame_queue))
    batch_thread.start()

    learns = 0
    while True:
        while len(thread_dict["batches"]) > 0:
            batch = thread_dict["batches"].pop()
            learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4])
            learns += 1

            shared_state_dict.load_state_dict(learner.policy_net.state_dict())

            if learns % save_every == 0:
                #learner.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                    thread_dict["frames_generated"],
                    learns,
                    epsilon.value,
                    np.mean(thread_dict["rewards"]),
                ))

            epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

        else:
            time.sleep(0.1)

    batch_thread.join()
    for p in generator_processes:
        p.join()