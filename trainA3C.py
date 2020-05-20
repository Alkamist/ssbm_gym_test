import time
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

from melee import Melee

os.environ["OMP_NUM_THREADS"] = "1"


options = dict(
    windows=True,
    render=False,
    speed=0,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

#WORKER_COUNT = mp.cpu_count()
WORKER_COUNT = 8

STATE_SIZE = 792
ACTION_SIZE = 30

EPISODE_STEPS = 3600
SYNC_GLOBAL_EVERY = 5
GAMMA = 0.9
MAX_EPISODES = 6000

#def calculate_reward(state, next_state):
#    return 1.0 if abs(next_state.players[0].x - 25.0) <= 5.0 else 0.0

def calculate_reward(state, next_state):
    def player_is_dying(player):
        return player.action_state <= 0xA

    def player_just_died(player_index):
        return (not player_is_dying(state.players[player_index])) and player_is_dying(next_state.players[player_index])

    reward = 0.0

    if player_just_died(1):
        reward = 1.0

    if player_just_died(0):
        reward = -1.0

    return reward


def numpy_to_tensor(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

def push_and_pull(optimizer, local_net, global_net, done, next_state, bs, ba, br, gamma):
    if done:
        v_s_ = 0.0 # terminal
    else:
        v_s_ = local_net.forward(numpy_to_tensor(next_state[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]: # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_net.compute_loss(
        numpy_to_tensor(np.vstack(bs)),
        numpy_to_tensor(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else numpy_to_tensor(np.vstack(ba)),
        numpy_to_tensor(np.array(buffer_v_target)[:, None])
    )

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    loss.backward()
    for local_parameter, global_parameter in zip(local_net.parameters(), global_net.parameters()):
        global_parameter._grad = local_parameter.grad
    optimizer.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())

def record(global_episode, global_episode_reward, episode_reward, result_queue, name):
    with global_episode.get_lock():
        global_episode.value += 1
    with global_episode_reward.get_lock():
        if global_episode_reward.value == 0.:
            global_episode_reward.value = episode_reward
        else:
            global_episode_reward.value = global_episode_reward.value * 0.99 + episode_reward * 0.01
    result_queue.put(global_episode_reward.value)
    print(
        name,
        "Ep:", global_episode.value,
        "| Ep_r: %.4f" % global_episode_reward.value,
    )


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Net, self).__init__()
        self.pi1 = nn.Linear(state_size, hidden_size)
        self.pi2 = nn.Linear(hidden_size, action_size)
        self.v1 = nn.Linear(state_size, hidden_size)
        self.v2 = nn.Linear(hidden_size, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, state):
        self.eval()
        logits, _ = self.forward(state)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def compute_loss(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, result_queue, worker_id):
        super(Worker, self).__init__()
        self.name = 'w%02i' % worker_id
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.result_queue = result_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.local_net = Net(STATE_SIZE, ACTION_SIZE)           # local network
        self.worker_id = worker_id
        self.melee = None

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EPISODES:
            if self.melee is None:
                self.melee = Melee(worker_id=self.worker_id, **options)

            state = self.melee.reset()
            state_embed = np.array(self.melee.embed_state())
            state_buffer = []
            action_buffer = []
            reward_buffer = []
            episode_reward = 0.0

            episode_step = 0
            while True:
                action = self.local_net.choose_action(numpy_to_tensor(state_embed[None, :]))
                next_state = self.melee.step(action)
                next_state_embed = np.array(self.melee.embed_state())
                r = calculate_reward(state, next_state)
                #done = episode_step >= EPISODE_STEPS
                done = r != 0.0
                reward = 1.0 if r >= 1.0 else 0.0

                if done:
                    reward = -1

                episode_reward += reward
                action_buffer.append(action)
                state_buffer.append(state_embed)
                reward_buffer.append(reward)

                # update global and assign to local net
                if total_step % SYNC_GLOBAL_EVERY == 0 or done:
                    # sync
                    push_and_pull(self.optimizer, self.local_net, self.global_net, done, next_state_embed, state_buffer, action_buffer, reward_buffer, GAMMA)
                    state_buffer = []
                    action_buffer = []
                    reward_buffer = []

                    if done:
                        record(self.global_episode, self.global_episode_reward, episode_reward, self.result_queue, self.name)
                        break

                state = deepcopy(next_state)
                state_embed = next_state_embed
                total_step += 1
                episode_step += 1

        self.result_queue.put(None)


if __name__ == "__main__":
    global_net = Net(STATE_SIZE, ACTION_SIZE)
    #global_net.load_state_dict(torch.load("checkpoints/agent.pth"))
    global_net.share_memory()
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.)
    result_queue = mp.Queue()

    # parallel training
    workers = [Worker(global_net, optimizer, global_episode, global_episode_reward, result_queue, i) for i in range(WORKER_COUNT)]
    [w.start() for w in workers]
    res = [] # record episode reward to plot
    start_time = time.time()
    save_count = 0
    while True:
        current_time = time.time()
        if current_time - start_time >= 60.0:
            torch.save(global_net.state_dict(), "checkpoints/" + str(save_count) + ".pth")
            start_time = current_time
            save_count += 1

        reward = result_queue.get()
        if reward is not None:
            res.append(reward)
        else:
            break
    [w.join() for w in workers]

    save_count += 1
    torch.save(global_net.state_dict(), "checkpoints/" + str(save_count) + ".pth")

#    import matplotlib.pyplot as plt
#    plt.plot(res)
#    plt.ylabel('Moving average ep reward')
#    plt.xlabel('Step')
#    plt.show()
