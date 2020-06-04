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
num_policy_workers = 4
batch_size = 64
learn_every = 64
save_every = 200

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 1000


def generate_random_frames(worker_id, frame_queue):
    env = MeleeEnv(worker_id=worker_id, **melee_options)

    states = env.reset()

    while True:
        #try:
        actions = [random.randrange(MeleeEnv.num_actions), random.randrange(MeleeEnv.num_actions)]

        step_env_with_timeout = timeout(5)(lambda : env.step(actions))
        next_states, rewards, dones, _ = step_env_with_timeout()

        for player_id in range(2):
            frame_queue.put((states[player_id],
                             actions[player_id],
                             rewards[player_id],
                             next_states[player_id],
                             dones[player_id]))

        states = deepcopy(next_states)

        #except KeyboardInterrupt:
        #    env.close()

        #except:
        #    states = env.reset()
        #    return


def generate_policy_frames(worker_id, shared_state_dict, frame_queue):
    random_action_chance = 0.005

    env = MeleeEnv(worker_id=worker_id, **melee_options)

    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)

    states = env.reset()

    with torch.no_grad():
        while True:
            policy_net.load_state_dict(shared_state_dict.state_dict())

            actions = []
            for player_id in range(2):
                if random.random() > random_action_chance:
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


def add_frames_to_buffer(replay_buffer, thread_dict, frame_queue):
    frames_generated = 0
    while True:
        frame = frame_queue.get()
        replay_buffer.add(*frame)
        frames_generated += 1
        if frames_generated % learn_every == 0:
            thread_dict["learns_allowed"] += 1



if __name__ == "__main__":
    reward_buffer = []

    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=0.0001)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    random_replay_buffer = ReplayBuffer(100000)
    policy_replay_buffer = ReplayBuffer(3600)

    random_queue = mp.Queue()
    random_process = mp.Process(target=generate_random_frames, args=(512, random_queue))
    random_process.start()

    policy_queue = mp.Queue()
    policy_processes = []
    for worker_id in range(num_policy_workers):
        p = mp.Process(target=generate_policy_frames, args=(worker_id, shared_state_dict, policy_queue))
        p.start()
        policy_processes.append(p)

    random_thread_dict = {"learns_allowed" : 0}
    random_buffer_thread = threading.Thread(target=add_frames_to_buffer, args=(random_replay_buffer, random_thread_dict, random_queue))
    random_buffer_thread.start()

    policy_thread_dict = {"learns_allowed" : 0}
    policy_buffer_thread = threading.Thread(target=add_frames_to_buffer, args=(policy_replay_buffer, policy_thread_dict, policy_queue))
    policy_buffer_thread.start()

    learn_iterations = 0
    learn_iterations_since_print = 0
    while True:
        while random_thread_dict["learns_allowed"] > 0:
            network.learn(random_replay_buffer)
            shared_state_dict.load_state_dict(network.policy_net.state_dict())
            random_thread_dict["learns_allowed"] -= 1
            learn_iterations_since_print += 1

        while policy_thread_dict["learns_allowed"] > 0:
            network.learn(policy_replay_buffer)
            shared_state_dict.load_state_dict(network.policy_net.state_dict())
            policy_thread_dict["learns_allowed"] -= 1
            learn_iterations_since_print += 1

        if learn_iterations_since_print >= 200:
            learn_iterations += learn_iterations_since_print
            learn_iterations_since_print = 0
            print("Learn Iterations: {} / Average Rewards: {:.4f}".format(learn_iterations, 1.0))
            #rewards = []

        time.sleep(0.1)

########    rewards = []
########    policy_frames_added = 0
########    learn_iterations = 0
########    while True:
########        policy_frame = policy_frame_queue.get()
########        rewards.append(policy_frame[2])
########        policy_replay_buffer.add(*policy_frame)
########        policy_frames_added += 1
########
########        if info_flags["random_learns_allowed"] > 0:
########            network.learn(random_replay_buffer)
########            shared_state_dict.load_state_dict(network.policy_net.state_dict())
########            info_flags["random_learns_allowed"] -= 1
########            learn_iterations += 1
########
########        if policy_frames_added % 8 == 0:
########            network.learn(policy_replay_buffer)
########            shared_state_dict.load_state_dict(network.policy_net.state_dict())
########            learn_iterations += 1
########
########        if policy_frames_added % 14400 == 0:
########            print("Learn Iterations: {} / Average Rewards: {:.4f}".format(learn_iterations, np.mean(rewards)))
########            rewards = []

    random_buffer_thread.join()
    random_process.join()

    policy_buffer_thread.join()
    for p in policy_processes:
        p.join()