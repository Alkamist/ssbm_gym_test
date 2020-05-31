import random
import threading
from copy import deepcopy

import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from replay_buffer import ReplayBuffer
from DQN import DQN


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
num_workers = 8
batch_size = 128
learn_every = 60
save_every = 500
replay_buffer_size = 250000
seed = 1


def generate_frames(worker_id, frame_queue):
    env = MeleeEnv(worker_id=worker_id, **melee_options)

    states = env.reset()

    while True:
        actions = [random.randrange(MeleeEnv.num_actions), 1]
        next_states, rewards, _, _ = env.step(actions)
        frame_queue.put((states[0], actions[0], next_states[0], rewards[0]))
        states = deepcopy(next_states)


def add_frames_to_replay_buffer(info_flags, replay_buffer, frame_queue):
    while True:
        info_flags["frames_generated"] += 1

        frame = frame_queue.get()
        replay_buffer.add(frame[0], frame[1], frame[2], frame[3])

        if info_flags["frames_generated"] % learn_every == 0:
            info_flags["learns_allowed"] += 1


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, device)
    #network.load("checkpoints/agent.pth")

    replay_buffer = ReplayBuffer(replay_buffer_size, device)

    frame_queue = mp.Queue()

    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, frame_queue))
        p.start()
        generator_processes.append(p)

    info_flags = {
        "frames_generated" : 0,
        "learns_allowed" : 0,
    }

    replay_buffer_frame_thread = threading.Thread(target=add_frames_to_replay_buffer, args=(info_flags, replay_buffer, frame_queue))
    replay_buffer_frame_thread.start()

    need_to_save = False
    learn_iterations = 0
    while True:
        if info_flags["learns_allowed"] > 0 and len(replay_buffer) > batch_size:
            network.learn(replay_buffer.sample(batch_size))
            learn_iterations += 1
            info_flags["learns_allowed"] -= 1
            need_to_save = True

        if learn_iterations % save_every == 0 and learn_iterations > 0 and need_to_save:
            print("Total Frames: {} / Learn Iterations: {}".format(info_flags["frames_generated"], learn_iterations))
            network.save("checkpoints/agent" + str(learn_iterations) + ".pth")
            need_to_save = False

    replay_buffer_frame_thread.join()
    for p in generator_processes:
        p.join()