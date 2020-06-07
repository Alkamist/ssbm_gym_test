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
num_workers = 1
trajectory_steps = 30
learning_rate = 0.0001
batch_size = 16

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 80


def generate_trajectories(worker_id, shared_state_dict, output_queue, epsilon, learner_is_ready):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    policy_net.eval()

    storage_buffer = StorageBuffer(3000)

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    total_reward = 0.0
    frames_since_send = 0
    with torch.no_grad():
        while True:
            policy_net.load_state_dict(shared_state_dict.state_dict())

            initial_lstm_state = policy_net.lstm_state
            if initial_lstm_state is not None:
                initial_lstm_state = (initial_lstm_state[0].cpu(), initial_lstm_state[1].cpu())

            trajectory_states = []
            trajectory_actions = []
            trajectory_rewards = []
            trajectory_next_states = []
            trajectory_dones = []

            for _ in range(trajectory_steps):
                frames_since_send += 1

                state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
                action = policy_net(state).max(2)[1].item()

                if action_repeat_count > 0:
                    action = action_to_repeat
                    action_repeat_count -= 1
                else:
                    if random.random() <= epsilon.value:
                        action = random.randrange(MeleeEnv.num_actions)
                        action_to_repeat = action
                        action_repeat_count = random.randrange(12)

                actions = [action, 0]
                next_states, rewards, dones, _ = env.step(actions)

                total_reward += rewards[0]

                trajectory_states.append(states[0])
                trajectory_actions.append(actions[0])
                trajectory_rewards.append(rewards[0])
                trajectory_next_states.append(next_states[0])
                trajectory_dones.append(dones[0])

                states = deepcopy(next_states)

            storage_buffer.add_item((trajectory_states,
                                     trajectory_actions,
                                     trajectory_rewards,
                                     trajectory_next_states,
                                     trajectory_dones,
                                     initial_lstm_state))

            if learner_is_ready.value and len(storage_buffer) > batch_size:
                learner_is_ready.value = False

                batch_of_trajectories = storage_buffer.sample_batch(batch_size)

                trajectory_of_state_batches = [[] for _ in range(trajectory_steps)]
                trajectory_of_action_batches = [[] for _ in range(trajectory_steps)]
                trajectory_of_reward_batches = [[] for _ in range(trajectory_steps)]
                trajectory_of_next_state_batches = [[] for _ in range(trajectory_steps)]
                trajectory_of_done_batches = [[] for _ in range(trajectory_steps)]
                batch_of_lstm_states = []

                for trajectory in batch_of_trajectories:
                    states, actions, rewards, next_states, dones, lstm_state = trajectory
                    batch_of_lstm_states.append(lstm_state)
                    for step in range(trajectory_steps):
                        trajectory_of_state_batches[step].append(states[step])
                        trajectory_of_action_batches[step].append(actions[step])
                        trajectory_of_reward_batches[step].append(rewards[step])
                        trajectory_of_next_state_batches[step].append(next_states[step])
                        trajectory_of_done_batches[step].append(dones[step])

                output_queue.put((trajectory_of_state_batches,
                                  trajectory_of_action_batches,
                                  trajectory_of_reward_batches,
                                  trajectory_of_next_state_batches,
                                  trajectory_of_done_batches,
                                  batch_of_lstm_states,
                                  total_reward / frames_since_send))

                total_reward = 0.0
                frames_since_send = 0


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    learner_is_ready = mp.Value('b', True)
    epsilon = mp.Value('d', 1.0)

    trajectory_queue = mp.Queue()
    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_trajectories, args=(worker_id, shared_state_dict, trajectory_queue, epsilon, learner_is_ready))
        p.start()
        generator_processes.append(p)

    learns = 0
    loops = 0
    while True:
        loops += 1

        learner_is_ready.value = True
        trajectory = trajectory_queue.get()

        network.learn(trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4], trajectory[5])
        learns += 1

        shared_state_dict.load_state_dict(network.policy_net.state_dict())

        if loops % 80 == 0:
            #network.save("checkpoints/agent" + str(learns) + ".pth")
            print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                loops * batch_size * trajectory_steps,
                learns,
                epsilon.value,
                trajectory[6],
            ))

        epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

    for p in generator_processes:
        p.join()