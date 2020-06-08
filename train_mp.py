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

num_workers = 4
learning_rate = 0.0001
batch_size = 16

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 700


def generate_frames(worker_id, shared_state_dict, frame_queue, epsilon, learner_is_ready):
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    policy_net.eval()

    storage_buffer = StorageBuffer(10000)

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    rewards_since_output = 0.0
    frames_since_output = 0
    while True:
        frames_since_output += 1

        policy_net.load_state_dict(shared_state_dict.state_dict())

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

        rewards_since_output += rewards[0]

        storage_buffer.add_item((states[0],
                                 actions[0],
                                 rewards[0],
                                 next_states[0],
                                 dones[0],
                                 policy_net.rnn_state[0].cpu().detach(),
                                 policy_net.rnn_state[1].cpu().detach()))

        states = deepcopy(next_states)

        if learner_is_ready.value and len(storage_buffer) > batch_size:
            batch = storage_buffer.sample_batch(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, rnn_state_batch, rnn_cell_batch = zip(*batch)

            state_batch = torch.tensor([state_batch], dtype=torch.float32)
            action_batch = torch.tensor([action_batch], dtype=torch.long).unsqueeze(2)
            reward_batch = torch.tensor([reward_batch], dtype=torch.float32)
            next_state_batch = torch.tensor([next_state_batch], dtype=torch.float32)
            done_batch = torch.tensor([done_batch], dtype=torch.float32)
            rnn_state_batch = (torch.cat(rnn_state_batch, dim=1), torch.cat(rnn_cell_batch, dim=1))

            frame_queue.put((state_batch,
                             action_batch,
                             reward_batch,
                             next_state_batch,
                             done_batch,
                             rnn_state_batch,
                             rewards_since_output / frames_since_output))

            rewards_since_output = 0.0
            frames_since_output = 0


if __name__ == "__main__":
    learner = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #learner.load("checkpoints/agent.pth")

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.load_state_dict(learner.policy_net.state_dict())
    shared_state_dict.share_memory()

    learner_is_ready = mp.Value('b', True)
    epsilon = mp.Value('d', epsilon_start)

    frame_queue = mp.Queue()
    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, shared_state_dict, frame_queue, epsilon, learner_is_ready))
        p.start()
        generator_processes.append(p)

    learns = 0
    while True:
        learner_is_ready.value = True
        batch = frame_queue.get()

        learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
        learns += 1

        shared_state_dict.load_state_dict(learner.policy_net.state_dict())

        if learns % 500 == 0:
            #network.save("checkpoints/agent" + str(learns) + ".pth")
            print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                learns * batch_size,
                learns,
                epsilon.value,
                batch[6],
            ))

        epsilon.value = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

    for p in generator_processes:
        p.join()