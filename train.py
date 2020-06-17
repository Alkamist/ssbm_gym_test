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
from replay_buffer import PrioritizedReplayBuffer as ReplayBuffer
#from replay_buffer import ReplayBuffer
from IQN import DQNLearner, DQN
#from DQN import DQNLearner, DQN


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

num_workers = 1
batch_size = 16
memory_size = 18000
save_every = 500

n_step_size = 5
hidden_size = 512
gamma = 0.997
learning_rate = 0.0001
grad_norm_clipping = 5.0
target_update_frequency= 2500
use_dueling_net = True

epsilon_start = 0.01
epsilon_end = 0.01
epsilon_decay = 100000


def generate_frames(worker_id, shared_state_dict, frame_queue, epsilon):
    policy_net = DQN(
        input_size=MeleeEnv.observation_size,
        output_size=MeleeEnv.num_actions,
        hidden_size=hidden_size,
        use_dueling_net=use_dueling_net,
    ).to(device)
    policy_net.eval()

    env = MeleeEnv(worker_id=worker_id, **melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    with torch.no_grad():
        while True:
            policy_net.load_state_dict(shared_state_dict.state_dict())

            if action_repeat_count > 0:
                action = action_to_repeat
                action_repeat_count -= 1
            else:
                if random.random() <= epsilon.value:
                    action = random.randrange(MeleeEnv.num_actions)
                    action_to_repeat = action
                    action_repeat_count = random.randrange(4)
                else:
                    state = torch.tensor([states[0]], dtype=torch.float32, device=device)
                    action = policy_net(state).max(1)[1].item()

            #if random.random() <= epsilon.value:
            #    action = random.randrange(MeleeEnv.num_actions)
            #else:
            #    state = torch.tensor([states[0]], dtype=torch.float32, device=device)
            #    action = policy_net(state).max(1)[1].item()

            actions = [action, 0]
            next_states, rewards, dones, score = env.step(actions)

            frame_queue.put((states[0],
                             action,
                             rewards[0],
                             next_states[0],
                             dones[0],
                             rewards[0]))

            states = deepcopy(next_states)


def prepare_batches(replay_buffer, thread_dict, frame_queue):
    while True:
        frame = frame_queue.get()

        replay_buffer.add(frame[0], frame[1], frame[2], frame[3], frame[4])

        thread_dict["frames_generated"] += 1
        thread_dict["score"].append(frame[5])

        if len(replay_buffer) > batch_size:
            (batch_of_frames), weights, indices = replay_buffer.sample_batch(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch_of_frames)

            thread_dict["batches"].append((
                torch.cat(state_batch, dim=0),
                torch.cat(action_batch, dim=0),
                torch.cat(reward_batch, dim=0),
                torch.cat(next_state_batch, dim=0),
                torch.cat(done_batch, dim=0),
                weights,
                indices,
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
        use_dueling_net=use_dueling_net,
    )
    if load_model is not None:
        learner.load(load_model)

    shared_state_dict = DQN(
        input_size=MeleeEnv.observation_size,
        output_size=MeleeEnv.num_actions,
        hidden_size=hidden_size,
        use_dueling_net=use_dueling_net,
    )
    shared_state_dict.load_state_dict(learner.policy_net.state_dict())
    shared_state_dict.share_memory()

    replay_buffer = ReplayBuffer(
        max_size=memory_size,
        n_step_size=n_step_size,
        gamma=gamma,
        device=device,
    )
    frame_queue = mp.Queue(maxsize=1)
    epsilon = mp.Value("d", epsilon_start)

    generator_processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=generate_frames, args=(worker_id, shared_state_dict, frame_queue, epsilon))
        p.start()
        generator_processes.append(p)

    thread_dict = {
        "batches" : deque(maxlen=8),
        "frames_generated" : 0,
        "score" : deque(maxlen=18000),
    }
    batch_thread = threading.Thread(target=prepare_batches, args=(replay_buffer, thread_dict, frame_queue))
    batch_thread.start()

    learns = 0
    while True:
        while len(thread_dict["batches"]) > 0:
            batch = thread_dict["batches"].pop()
            errors = learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
            replay_buffer.update_priorities_from_errors(batch[6], errors)
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