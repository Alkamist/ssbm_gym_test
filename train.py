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
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0001
batch_size = 16
memory_size = 100000
save_every = 4000

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 10000


def generate_frames(worker_id, learner, thread_dict):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions, device=device)
    policy_net.eval()

    storage_buffer = StorageBuffer(memory_size)

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    while True:
        thread_dict["frames_generated"] += 1

        policy_net.load_state_dict(learner.policy_net.state_dict())

        #for goal_number in range(MeleeEnv.num_goals):
        state_that_causes_action = torch.tensor([[states[0][0]]], dtype=torch.float32, device=device)
        action = policy_net(state_that_causes_action).max(2)[1]

        if random.random() <= thread_dict["epsilon"]:
            action = torch.tensor([[random.randrange(MeleeEnv.num_actions)]], dtype=torch.long, device=device)

        actions = [action.item(), 0]
        next_states, rewards, dones, _ = env.step(actions)

        for goal_number in range(MeleeEnv.num_goals):
            storage_buffer.add_item(
                (torch.tensor([[states[goal_number][0]]], dtype=torch.float32, device=device),
                 action.unsqueeze(2),
                 torch.tensor([[rewards[goal_number][0]]], dtype=torch.float32, device=device),
                 torch.tensor([[next_states[goal_number][0]]], dtype=torch.float32, device=device),
                 torch.tensor([[dones[goal_number][0]]], dtype=torch.float32, device=device))
            )

        states = deepcopy(next_states)

        thread_dict["rewards"].append(rewards[0])

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

    generator_thread_dict = {
        "batches" : deque(maxlen=8),
        "epsilon" : epsilon_start,
        "frames_generated" : 0,
        "rewards" : deque(maxlen=3600),
    }
    generator_thread = threading.Thread(target=generate_frames, args=(0, learner, generator_thread_dict))
    generator_thread.start()

    learns = 0
    while True:
        while len(generator_thread_dict["batches"]) > 0:
            batch = generator_thread_dict["batches"].pop()
            learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4])
            learns += 1

            if learns % save_every == 0:
                learner.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                    generator_thread_dict["frames_generated"],
                    learns,
                    generator_thread_dict["epsilon"],
                    np.mean(generator_thread_dict["rewards"]) * 1000.0,
                ))

            generator_thread_dict["epsilon"] = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

        else:
            time.sleep(0.1)

    generator_thread.join()