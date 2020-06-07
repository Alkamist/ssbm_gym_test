import time
import math
import random
from copy import deepcopy

import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
#from replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
from replay_buffer import ReplayBuffer as ReplayBuffer
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

#learning_rate = 3e-5
num_workers = 2
learning_rate = 0.0001
batch_size = 16
print_every = 5000

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 20000


def generate_frames(worker_id, shared_state_dict, frame_queue, epsilon):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    policy_net.eval()

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    try:
        with torch.no_grad():
            while True:
                policy_net.load_state_dict(shared_state_dict.state_dict())

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

                frame_queue.put((states[0],
                                 actions[0],
                                 rewards[0],
                                 next_states[0],
                                 dones[0]))

                states = deepcopy(next_states)

    except KeyboardInterrupt:
        env.close()
        frame_queue.put(0)


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #network.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(network.policy_net.state_dict())
    shared_state_dict.share_memory()

    replay_buffer = ReplayBuffer(10000)

    epsilon = mp.Value('d', epsilon_start)

    frame_queue = mp.Queue()
    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, shared_state_dict, frame_queue, epsilon))
        p.start()
        generator_processes.append(p)

    should_terminate = False
    learns = 0
    learns_allowed = 0
    total_rewards = 0.0
    frames_generated = 0
    frames_since_print = 0
    while True:
        while frame_queue.qsize() > 8:
            frame = frame_queue.get()

            if frame == 0:
                should_terminate = True
                break

            replay_buffer.add(*frame)
            frames_generated += 1
            frames_since_print += 1
            learns_allowed += 1
            total_rewards += frame[2]

        if should_terminate:
            break

        if learns_allowed > 0:
            network.learn(replay_buffer)
            shared_state_dict.load_state_dict(network.policy_net.state_dict())
            learns_allowed -= 1
            learns += 1

            if learns % print_every == 0:
                network.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Extra Frames: {} / Average Reward: {:.4f}".format(
                    frames_generated,
                    learns,
                    epsilon.value,
                    frame_queue.qsize(),
                    total_rewards / frames_since_print,
                ))
                frames_since_print = 0
                total_rewards = 0

            epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

    for p in generator_processes:
        p.terminate()
    for p in generator_processes:
        p.join()