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
trajectory_steps = 60
save_every = 100

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 300


def generate_trajectories(worker_id, learner, thread_dict):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions, device=device)
    policy_net.eval()

    storage_buffer = StorageBuffer(960)

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    while True:
        policy_net.load_state_dict(learner.policy_net.state_dict())
        initial_rnn_state = (policy_net.rnn_state[0].detach(), policy_net.rnn_state[1].detach())

        state_trajectory = []
        action_trajectory = []
        reward_trajectory = []
        next_state_trajectory = []
        done_trajectory = []

        for _ in range(trajectory_steps):
            thread_dict["frames_generated"] += 1

            state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
            action = policy_net(state).max(2)[1]

            if action_repeat_count > 0:
                action = action_to_repeat
                action_repeat_count -= 1
            else:
                if random.random() <= thread_dict["epsilon"]:
                    action = torch.tensor([[random.randrange(MeleeEnv.num_actions)]], dtype=torch.long, device=device)
                    action_to_repeat = action
                    action_repeat_count = random.randrange(12)

            actions = [action.item(), 0]
            next_states, rewards, dones, _ = env.step(actions)

            thread_dict["rewards"].append(rewards[0])

            state_trajectory.append(state)
            action_trajectory.append(action.unsqueeze(2))
            reward_trajectory.append(torch.tensor([[rewards[0]]], dtype=torch.float32, device=device))
            next_state_trajectory.append(torch.tensor([[next_states[0]]], dtype=torch.float32, device=device))
            done_trajectory.append(torch.tensor([[dones[0]]], dtype=torch.float32, device=device))

            states = deepcopy(next_states)

        storage_buffer.add_item((torch.cat(state_trajectory, dim=0),
                                 torch.cat(action_trajectory, dim=0),
                                 torch.cat(reward_trajectory, dim=0),
                                 torch.cat(next_state_trajectory, dim=0),
                                 torch.cat(done_trajectory, dim=0),
                                 initial_rnn_state))

        if len(storage_buffer) > batch_size:
            batch_of_trajectories = storage_buffer.sample_batch(batch_size)

            output = []
            for index in range(5):
                output.append([])
                for trajectory in batch_of_trajectories:
                    output[index].append(trajectory[index])
                output[index] = torch.cat(output[index], dim=1)

            rnn_states = []
            rnn_cell_states = []
            for trajectory in batch_of_trajectories:
                full_state = trajectory[5]
                rnn_states.append(full_state[0])
                rnn_cell_states.append(full_state[1])

            batched_rnn_state = (torch.cat(rnn_states, dim=1), torch.cat(rnn_cell_states, dim=1))
            output.append(batched_rnn_state)

            thread_dict["batches"].append(output)


if __name__ == "__main__":
    learner = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate, target_update_frequency=2500//trajectory_steps)
    #learner.load("checkpoints/agent.pth")

    generator_thread_dict = {
        "batches" : deque(maxlen=8),
        "epsilon" : epsilon_start,
        "frames_generated" : 0,
        "rewards" : deque(maxlen=3600),
    }
    generator_thread = threading.Thread(target=generate_trajectories, args=(0, learner, generator_thread_dict))
    generator_thread.start()

    learns = 0
    while True:
        while len(generator_thread_dict["batches"]) > 0:
            batch = generator_thread_dict["batches"].pop()
            learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
            learns += 1

            if learns % save_every == 0:
                #learner.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                    generator_thread_dict["frames_generated"],
                    learns,
                    generator_thread_dict["epsilon"],
                    np.mean(generator_thread_dict["rewards"]),
                ))

            generator_thread_dict["epsilon"] = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

        else:
            time.sleep(0.1)

    generator_thread.join()