import time
import math
import random
import threading
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from replay_buffer import ReplayBuffer
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
num_workers = 4
batch_size = 64
learn_every = 32
save_every = 1000
replay_buffer_size = 100000

epsilon_start = 1.0
epsilon_end = 1.0
epsilon_decay = 10000


def test_policy(policy, testing_steps):
    env = MeleeEnv(worker_id=1024, **melee_options)

    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32, device=device)

    total_rewards = 0.0

    for _ in range(testing_steps):
        try:
            actions = policy(states).max(1)[1]

            step_env_with_timeout = timeout(5)(lambda : env.step(actions))
            states, rewards, _, _ = step_env_with_timeout()
            states = torch.tensor(states, dtype=torch.float32, device=device)

            total_rewards += rewards[0]

        except KeyboardInterrupt:
            env.close()

        except:
            env.close()
            return 0.0

    env.close()
    return total_rewards / testing_steps


def generate_frames(worker_id, frame_queue):
    env = MeleeEnv(worker_id=worker_id, **melee_options)

    states = env.reset()

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
    while True:
        frame = frame_queue.get()

        if isinstance(frame, tuple):
            replay_buffer.add(frame[0],
                              frame[1],
                              frame[2],
                              frame[3])
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
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, device)
    #network.load("checkpoints/agent.pth")

    replay_buffer = ReplayBuffer(replay_buffer_size, device)

    frame_queue = mp.Queue()

    generator_processes = []
    for worker_id in range(num_workers):
        generator_processes.append(create_worker(worker_id, frame_queue))

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

    need_to_save = False
    learn_iterations = 0
    while True:
        if info_flags["dolphin_crashed"] is not None:
            print("A dolphin instance crashed.")
            generator_processes[info_flags["dolphin_crashed"]].terminate()
            generator_processes[info_flags["dolphin_crashed"]] = create_worker(info_flags["dolphin_crashed"], frame_queue)
            info_flags["dolphin_crashed"] = None

        should_learn = info_flags["learns_allowed"] > 0 and len(replay_buffer) > batch_size

        if should_learn:
            network.learn(replay_buffer.sample(batch_size))

            learn_iterations += 1
            info_flags["learns_allowed"] -= 1
            need_to_save = True

        if learn_iterations % save_every == 0 and learn_iterations > 0 and need_to_save:
            network.save("checkpoints/agent" + str(learn_iterations) + ".pth")
            need_to_save = False
            average_reward = test_policy(network.policy_net, 3600)
            print("Total Frames: {} / Learn Iterations: {} / Average Reward: {:.3f}".format(
                info_flags["frames_generated"],
                learn_iterations,
                average_reward,
            ))

    replay_buffer_frame_thread.join()
    for p in generator_processes:
        p.join()