import time
import math
import random
import threading
from copy import deepcopy

import torch
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
num_workers = 8
batch_size = 128
learn_every = 64
save_every = 1000
replay_buffer_size = 100000

epsilon_start = 1.0
epsilon_end = 1.0
epsilon_decay = 10000


def select_actions(policy, states, epsilon):
    with torch.no_grad():
        if random.random() > epsilon:
            return policy(states).max(1)[1]
        else:
            return torch.tensor([random.randrange(MeleeEnv.num_actions) for _ in range(2)], device=device, dtype=torch.long)


def generate_frames(worker_id, shared_state_dict, frame_queue, epsilon):
    #policy = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device)
    #policy.load_state_dict(shared_state_dict.state_dict())

    env = MeleeEnv(worker_id=worker_id, **melee_options)

    states = env.reset()
    #states = torch.tensor(states, dtype=torch.float32, device=device)

    #frame_count = 0
    while True:
        try:
            #frame_count += 1
            #if frame_count % 600 == 0:
            #    policy.load_state_dict(shared_state_dict.state_dict())

            actions = [random.randrange(MeleeEnv.num_actions), random.randrange(MeleeEnv.num_actions)]
            #actions = select_actions(policy, states, epsilon.value)

            step_env_with_timeout = timeout(5)(lambda : env.step(actions))
            next_states, rewards, _, _ = step_env_with_timeout()

            #next_states = torch.tensor(next_states, dtype=torch.float32, device=device)

            for player_id in range(2):
                frame_queue.put((states[player_id],
                                 actions[player_id],
                                 next_states[player_id],
                                 rewards[player_id]))

#            for player_id in range(2):
#                frame_queue.put((states[player_id].cpu().numpy(),
#                                 actions[player_id].item(),
#                                 next_states[player_id].cpu().numpy(),
#                                 rewards[player_id]))

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


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, device)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    replay_buffer = ReplayBuffer(replay_buffer_size, device)

    frame_queue = mp.Queue()
    epsilon = mp.Value("d", epsilon_start)

    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, shared_state_dict, frame_queue, epsilon))
        p.start()
        generator_processes.append(p)

    info_flags = {
        "frames_generated" : 0,
        "learns_allowed" : 0,
        "dolphin_crashed" : None,
    }

    replay_buffer_frame_thread = threading.Thread(target=add_frames_to_replay_buffer, args=(info_flags, replay_buffer, frame_queue))
    replay_buffer_frame_thread.start()

    need_to_save = False
    learn_iterations = 0
    while True:
        if info_flags["dolphin_crashed"] is not None:
            print("A dolphin instance crashed.")
            generator_processes[info_flags["dolphin_crashed"]].terminate()
            p = mp.Process(target=generate_frames, args=(info_flags["dolphin_crashed"], shared_state_dict, frame_queue, epsilon))
            p.start()
            generator_processes[info_flags["dolphin_crashed"]] = p
            info_flags["dolphin_crashed"] = None

        should_learn = info_flags["learns_allowed"] > 0 and len(replay_buffer) > batch_size

        if should_learn:
            network.learn(replay_buffer.sample(batch_size))

            epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learn_iterations / epsilon_decay)
            shared_state_dict.load_state_dict(network.policy_net.state_dict())

            learn_iterations += 1
            info_flags["learns_allowed"] -= 1
            need_to_save = True

        if learn_iterations % save_every == 0 and learn_iterations > 0 and need_to_save:
            print("Total Frames: {} / Learn Iterations: {} / Epsilon: {:.3f}".format(
                info_flags["frames_generated"],
                learn_iterations,
                epsilon.value,
            ))
            network.save("checkpoints/agent" + str(learn_iterations) + ".pth")
            need_to_save = False

    replay_buffer_frame_thread.join()
    for p in generator_processes:
        p.join()