import time
import math
import random
import threading
from copy import deepcopy

import torch

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

learning_rate = 0.0001
batch_size = 16

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 700


def generate_frames(worker_id, learner, thread_dict):
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

        policy_net.load_state_dict(learner.policy_net.state_dict())

        state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
        action = policy_net(state).max(2)[1]

        if action_repeat_count > 0:
            action = action_to_repeat
            action_repeat_count -= 1
        else:
            if random.random() <= thread_dict["epsilon"]:
                action = torch.tensor([[random.randrange(MeleeEnv.num_actions)]], dtype=torch.long, device=device)
                action_to_repeat = action
                action_repeat_count = random.randrange(12)

        actions = [action.item(), 0]
        next_states, rewards, dones, _ = env.step(actions)

        rewards_since_output += rewards[0]

        storage_buffer.add_item((state,
                                 action,
                                 torch.tensor([[rewards[0]]], dtype=torch.float32, device=device),
                                 torch.tensor([[next_states[0]]], dtype=torch.float32, device=device),
                                 torch.tensor([[dones[0]]], dtype=torch.float32, device=device),
                                 policy_net.rnn_state[0].detach(),
                                 policy_net.rnn_state[1].detach()))

        states = deepcopy(next_states)

        if len(storage_buffer) > batch_size:
            batch = storage_buffer.sample_batch(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, rnn_state_batch, rnn_cell_batch = zip(*batch)

            generator_thread_dict["batch"] = (
                torch.cat(state_batch, dim=1),
                torch.cat(action_batch, dim=1).unsqueeze(2),
                torch.cat(reward_batch, dim=1),
                torch.cat(next_state_batch, dim=1),
                torch.cat(done_batch, dim=1),
                (torch.cat(rnn_state_batch, dim=1), torch.cat(rnn_cell_batch, dim=1)),
                rewards_since_output / frames_since_output,
            )

            rewards_since_output = 0.0
            frames_since_output = 0


if __name__ == "__main__":
    learner = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #learner.load("checkpoints/agent.pth")

    generator_thread_dict = {
        "batch" : None,
        "epsilon" : epsilon_start,
    }
    generator_thread = threading.Thread(target=generate_frames, args=(0, learner, generator_thread_dict))
    generator_thread.start()

    learns = 0
    while True:
        batch = generator_thread_dict["batch"]

        if batch is None:
            time.sleep(0.01)

        else:
            learner.learn(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5])
            learns += 1

            if learns % 500 == 0:
                #network.save("checkpoints/agent" + str(learns) + ".pth")
                print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                    learns * batch_size,
                    learns,
                    generator_thread_dict["epsilon"],
                    batch[6],
                ))

            generator_thread_dict["epsilon"] = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * learns / epsilon_decay)

    generator_thread.join()