import time
import math
import random
import threading
from copy import deepcopy
from collections import deque

import numpy as np
import torch

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

learning_rate = 0.0001
batch_size = 16
memory_size = 100000
save_every = 2000

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 4000


def generate_frames(worker_id, learner, storage_buffer, thread_dict):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions, device=device)
    policy_net.eval()

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    with torch.no_grad():
        while True:
            thread_dict["frames_generated"] += 1

            policy_net.load_state_dict(learner.policy_net.state_dict())

            state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
            #action = policy_net(state).max(2)[1]

            if random.random() <= thread_dict["epsilon"]:
                action = torch.tensor([[random.randrange(MeleeEnv.num_actions)]], dtype=torch.long, device=device)
            else:
                action = policy_net(state).max(2)[1]

            actions = [action.item(), 0]
            next_states, rewards, dones, _ = env.step(actions)

            storage_buffer.add_item((state,
                                    action.unsqueeze(2),
                                    torch.tensor([[rewards[0]]], dtype=torch.float32, device=device),
                                    torch.tensor([[next_states[0]]], dtype=torch.float32, device=device),
                                    torch.tensor([[dones[0]]], dtype=torch.float32, device=device)))

            states = deepcopy(next_states)

            thread_dict["rewards"].append(rewards[0])


def prepare_batches(storage_buffer, thread_dict):
    while True:
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
    learner = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate, target_update_frequency=2500)
    #learner.load("checkpoints/agent.pth")

    storage_buffer = StorageBuffer(memory_size)

    thread_dict = {
        "batches" : deque(maxlen=8),
        "epsilon" : epsilon_start,
        "frames_generated" : 0,
        "rewards" : deque(maxlen=3600),
    }
    generator_thread = threading.Thread(target=generate_frames, args=(0, learner, storage_buffer, thread_dict))
    generator_thread.start()

    batch_thread = threading.Thread(target=prepare_batches, args=(storage_buffer, thread_dict))
    batch_thread.start()

    learns = 0
    while True:
        while len(thread_dict["batches"]) > 0:
            batch = thread_dict["batches"].pop()
            learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4])
            learns += 1

            if learns % save_every == 0:
                #learner.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                    thread_dict["frames_generated"],
                    learns,
                    thread_dict["epsilon"],
                    np.mean(thread_dict["rewards"]),
                ))

            thread_dict["epsilon"] = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

        else:
            time.sleep(0.1)

    batch_thread.join()
    generator_thread.join()