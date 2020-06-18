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
from action_selector import ActionSelector
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
#from IQN import DQNLearner, DQN
from DQN import DQNLearner, DQN


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

#load_model = "checkpoints/agent.pth"
load_model = None

trajectory_steps = 5

num_workers = 1
batch_size = 16
memory_size = 18000 // trajectory_steps
save_every = 500 // trajectory_steps

use_action_repeats = False
action_max_repeats = 4
n_step_size = 5
hidden_size = 512
gamma = 0.997
learning_rate = 0.0001
grad_norm_clipping = 5.0
target_update_frequency = 2500 // trajectory_steps
use_per = True
use_dueling_net = True
use_rnn = True

epsilon_start = 0.01
epsilon_end = 0.01
epsilon_decay = 100000 // trajectory_steps


def generate_trajectories(worker_id, shared_state_dict, output_queue, epsilon):
    policy_net = DQN(
        input_size=MeleeEnv.observation_size,
        output_size=MeleeEnv.num_actions,
        hidden_size=hidden_size,
        use_rnn=use_rnn,
        use_dueling_net=use_dueling_net,
        device=device,
    )
    policy_net.eval()

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_selector = ActionSelector(
        num_actions=MeleeEnv.num_actions,
        use_repeating=use_action_repeats,
        max_repeats=action_max_repeats,
    )

    with torch.no_grad():
        while True:
            policy_net.load_state_dict(shared_state_dict.state_dict())

            if use_rnn:
                initial_rnn_state = policy_net.rnn_state[0].detach().cpu()
                initial_rnn_cell = policy_net.rnn_state[1].detach().cpu()
            else:
                initial_rnn_state = None
                initial_rnn_cell = None

            state_trajectory = []
            action_trajectory = []
            reward_trajectory = []
            next_state_trajectory = []
            done_trajectory = []

            trajectory_score = 0

            for _ in range(trajectory_steps):
                state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
                action = policy_net(state).max(2)[1].item()

                random_action, was_random = action_selector.select_action(epsilon.value)
                if was_random:
                    action = random_action

                actions = [action, 0]
                next_states, rewards, dones, score = env.step(actions)

                #trajectory_score += score
                trajectory_score += rewards[0]

                state_trajectory.append(states[0])
                action_trajectory.append(action)
                reward_trajectory.append(rewards[0])
                next_state_trajectory.append(next_states[0])
                done_trajectory.append(dones[0])

                states = deepcopy(next_states)

            output_queue.put((state_trajectory,
                              action_trajectory,
                              reward_trajectory,
                              next_state_trajectory,
                              done_trajectory,
                              initial_rnn_state,
                              initial_rnn_cell,
                              trajectory_score))


def prepare_batches(replay_buffer, thread_dict, trajectory_queue):
    while True:
        trajectory = trajectory_queue.get()
        states, actions, rewards, next_states, dones, rnn_state, rnn_cell, score = trajectory
        replay_buffer.add_trajectory(states,
                                     actions,
                                     rewards,
                                     next_states,
                                     dones,
                                     rnn_state.to(device) if use_rnn else None,
                                     rnn_cell.to(device) if use_rnn else None)

        thread_dict["frames_generated"] += trajectory_steps
        thread_dict["score"].append(score)

        if len(replay_buffer) > batch_size:
            if use_per:
                (batch_of_trajectories), weights, indices = replay_buffer.sample_batch(batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, rnn_state_batch, rnn_cell_batch = zip(*batch_of_trajectories)

                thread_dict["batches"].append((
                    torch.cat(state_batch, dim=1),
                    torch.cat(action_batch, dim=1),
                    torch.cat(reward_batch, dim=1),
                    torch.cat(next_state_batch, dim=1),
                    torch.cat(done_batch, dim=1),
                    torch.cat(rnn_state_batch, dim=1) if use_rnn else None,
                    torch.cat(rnn_cell_batch, dim=1) if use_rnn else None,
                    weights,
                    indices,
                ))
            else:
                batch_of_trajectories = replay_buffer.sample_batch(batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, rnn_state_batch, rnn_cell_batch = zip(*batch_of_trajectories)

                thread_dict["batches"].append((
                    torch.cat(state_batch, dim=1),
                    torch.cat(action_batch, dim=1),
                    torch.cat(reward_batch, dim=1),
                    torch.cat(next_state_batch, dim=1),
                    torch.cat(done_batch, dim=1),
                    torch.cat(rnn_state_batch, dim=1) if use_rnn else None,
                    torch.cat(rnn_cell_batch, dim=1) if use_rnn else None,
                    None,
                ))


if __name__ == "__main__":
    learner = DQNLearner(
        state_size=MeleeEnv.observation_size,
        action_size=MeleeEnv.num_actions,
        hidden_size=hidden_size,
        batch_size=batch_size,
        n_step_size=n_step_size,
        device=device,
        learning_rate=learning_rate,
        gamma=gamma,
        grad_norm_clipping=grad_norm_clipping,
        target_update_frequency=target_update_frequency,
        use_rnn=use_rnn,
        use_dueling_net=use_dueling_net,
    )
    if load_model is not None:
        learner.load(load_model)

    shared_state_dict = DQN(
        input_size=MeleeEnv.observation_size,
        output_size=MeleeEnv.num_actions,
        hidden_size=hidden_size,
        use_rnn=use_rnn,
        use_dueling_net=use_dueling_net,
        device="cpu",
    )
    shared_state_dict.load_state_dict(learner.policy_net.state_dict())
    shared_state_dict.share_memory()

    if use_per:
        _ReplayBuffer = PrioritizedReplayBuffer
    else:
        _ReplayBuffer = ReplayBuffer
    replay_buffer = _ReplayBuffer(
        max_size=memory_size,
        n_step_size=n_step_size,
        gamma=gamma,
        device=device,
    )
    trajectory_queue = mp.Queue(maxsize=1)
    epsilon = mp.Value("d", epsilon_start)

    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_trajectories, args=(worker_id, shared_state_dict, trajectory_queue, epsilon))
        p.start()
        generator_processes.append(p)

    thread_dict = {
        "batches" : deque(maxlen=8),
        "frames_generated" : 0,
        "score" : deque(maxlen=3600),
    }
    batch_thread = threading.Thread(target=prepare_batches, args=(replay_buffer, thread_dict, trajectory_queue))
    batch_thread.start()

    learns = 0
    while True:
        while len(thread_dict["batches"]) > 0:
            batch = thread_dict["batches"].pop()
            errors = learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7])
            if use_per:
                replay_buffer.update_priorities_from_errors(batch[8], errors)
            learns += 1

            shared_state_dict.load_state_dict(learner.policy_net.state_dict())

            if learns % save_every == 0:
                #learner.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Score {:.4f}".format(
                    thread_dict["frames_generated"],
                    learns,
                    epsilon.value,
                    np.mean(thread_dict["score"]),
                    #np.sum(thread_dict["score"]),
                ))

            epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

        else:
            time.sleep(0.1)

    batch_thread.join()
    for p in generator_processes:
        p.join()