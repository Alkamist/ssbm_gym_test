import math
import random
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
epsilon_decay = 5000


if __name__ == "__main__":
    learner = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, batch_size, device, lr=learning_rate)
    #learner.load("checkpoints/agent.pth")

    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    policy_net.eval()

    storage_buffer = StorageBuffer(10000)

    env = MeleeEnv(**melee_options)
    states = env.reset()

    action_to_repeat = 0
    action_repeat_count = 0

    epsilon = epsilon_start

    learns = 0
    total_rewards = 0
    total_frames = 0
    frames_since_print = 0
    while True:
        total_frames += 1
        frames_since_print += 1

        policy_net.load_state_dict(learner.policy_net.state_dict())

        state = torch.tensor([[states[0]]], dtype=torch.float32, device=device)
        action = policy_net(state).max(2)[1].item()
        if action_repeat_count > 0:
            action = action_to_repeat
            action_repeat_count -= 1
        else:
            if random.random() <= epsilon:
                action = random.randrange(MeleeEnv.num_actions)
                action_to_repeat = action
                action_repeat_count = random.randrange(12)

        actions = [action, 0]
        next_states, rewards, dones, _ = env.step(actions)

        total_rewards += rewards[0]

        storage_buffer.add_item((states[0],
                                 actions[0],
                                 rewards[0],
                                 next_states[0],
                                 dones[0],
                                 policy_net.lstm_state[0].detach(),
                                 policy_net.lstm_state[1].detach()))

        states = deepcopy(next_states)

        if total_frames % 16 == 0 and len(storage_buffer) > batch_size:
            batch = storage_buffer.sample_batch(batch_size)
            learner.learn(*zip(*batch))
            learns += 1

        if frames_since_print >= 3600:
            #learner.save("checkpoints/agent" + str(learns) + ".pth")
            print("Frames: {} / Learns: {} / Epsilon: {:.2f} / Average Reward {:.4f}".format(
                total_frames,
                learns,
                epsilon,
                total_rewards / frames_since_print,
            ))
            total_rewards = 0
            frames_since_print = 0

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * total_frames / epsilon_decay)