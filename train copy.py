import time
import math
import random
from copy import deepcopy

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

#learning_rate = 3e-5
num_workers = 4
trajectory_steps = 60
learning_rate = 0.0001
batch_size = 16

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500


def generate_trajectories(worker_id, shared_state_dict, trajectory_queue, epsilon):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    policy_net.eval()

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    with torch.no_grad():
        while True:
            policy_net.load_state_dict(shared_state_dict.state_dict())

            trajectory_states = []
            trajectory_actions = []
            trajectory_rewards = []
            trajectory_next_states = []
            trajectory_dones = []

            for _ in range(trajectory_steps):
                if action_repeat_count > 0:
                    action = action_to_repeat
                    action_repeat_count -= 1
                else:
                    if random.random() > epsilon.value:
                        state = torch.tensor(states[0], dtype=torch.float32, device=device).unsqueeze(0)
                        action = policy_net(state).max(1)[1].item()
                    else:
                        action = random.randrange(MeleeEnv.num_actions)
                        action_to_repeat = action
                        action_repeat_count = random.randrange(12)

                actions = [action, 0]
                next_states, rewards, dones, _ = env.step(actions)

                trajectory_states.append(states[0])
                trajectory_actions.append(actions[0])
                trajectory_rewards.append(rewards[0])
                trajectory_next_states.append(next_states[0])
                trajectory_dones.append(dones[0])

                states = deepcopy(next_states)

            trajectory_queue.put((trajectory_states,
                                  trajectory_actions,
                                  trajectory_rewards,
                                  trajectory_next_states,
                                  trajectory_dones))


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    storage_buffer = StorageBuffer(3000)

    epsilon = mp.Value('d', epsilon_start)

    trajectory_queue = mp.Queue()
    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_trajectories, args=(worker_id, shared_state_dict, trajectory_queue, epsilon))
        p.start()
        generator_processes.append(p)

    learns = 0
    learns_allowed = 0
    total_rewards = 0.0
    trajectories_generated = 0
    trajectories_since_print = 0
    while True:
        while trajectory_queue.qsize() > 0:
            trajectory = trajectory_queue.get()
            storage_buffer.add_item(trajectory)
            trajectories_generated += 1
            trajectories_since_print += 1
            learns_allowed += 1

            for reward in trajectory[2]:
                total_rewards += reward

        if (learns_allowed > 0) and (len(storage_buffer) > batch_size):
            batch_of_trajectories = storage_buffer.sample_batch(batch_size)

            trajectory_of_state_batches = [[] for _ in range(trajectory_steps)]
            trajectory_of_action_batches = [[] for _ in range(trajectory_steps)]
            trajectory_of_reward_batches = [[] for _ in range(trajectory_steps)]
            trajectory_of_next_state_batches = [[] for _ in range(trajectory_steps)]
            trajectory_of_done_batches = [[] for _ in range(trajectory_steps)]
            for trajectory in batch_of_trajectories:
                states, actions, rewards, next_states, dones = trajectory
                for step in range(trajectory_steps):
                    trajectory_of_state_batches[step].append(states[step])
                    trajectory_of_action_batches[step].append(actions[step])
                    trajectory_of_reward_batches[step].append(rewards[step])
                    trajectory_of_next_state_batches[step].append(next_states[step])
                    trajectory_of_done_batches[step].append(dones[step])

            for step in range(trajectory_steps):
                network.learn(trajectory_of_state_batches[step],
                              trajectory_of_action_batches[step],
                              trajectory_of_reward_batches[step],
                              trajectory_of_next_state_batches[step],
                              trajectory_of_done_batches[step])
                learns_allowed -= 1
                learns += 1

            shared_state_dict.load_state_dict(network.policy_net.state_dict())

            if trajectories_since_print > 60:
                #network.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward: {:.4f}".format(
                    trajectories_generated * trajectory_steps,
                    learns,
                    epsilon.value,
                    total_rewards / (trajectories_since_print * trajectory_steps),
                ))
                trajectories_since_print = 0
                total_rewards = 0

            epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

    for p in generator_processes:
        p.join()