import os
import time
import queue
from copy import deepcopy
from collections import deque

import torch
import torch.optim as optim

from models import Policy

class Learner(object):
    def __init__(self, observation_size, num_actions, lr, c_hat, rho_hat, gamma,
                 value_loss_coef, entropy_coef, max_grad_norm, seed, max_batch_repeat,
                 episode_steps, queue_batch, shared_state_dict, device):
        self.c_hat = c_hat
        self.rho_hat = rho_hat
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.device = device
        self.queue_batch = queue_batch
        self.shared_state_dict = shared_state_dict
        self.max_batch_repeat = max_batch_repeat
        self.policy = Policy(observation_size, num_actions).to(self.device)
        self.policy.train()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.update_state_dict()

    def update_state_dict(self):
        self.shared_state_dict.load_state_dict(self.policy.state_dict())

    def learning(self):
        torch.manual_seed(self.seed)
        c_hat = self.c_hat
        rho_hat = self.rho_hat

        observations, actions, mu_log_probs, rewards = self.queue_batch.get(block=True)

        #i = 0
        batch_iter = 0
        #t = time.perf_counter()
        while True:
            try:
                # Retrain on previous batch if the next one is not ready yet
                block = batch_iter >= self.max_batch_repeat
                observations, actions, mu_log_probs, rewards = self.queue_batch.get(block=block)
                batch_iter = 0
            except queue.Empty:
                pass

            observations = observations.to(self.device)
            actions = actions.to(self.device)
            mu_log_probs = mu_log_probs.to(self.device)
            rewards = rewards.to(self.device)

            batch_iter += 1
            self.optimizer.zero_grad()

            values, pi_log_probs, entropy = self.policy.evaluate_actions(observations, actions)

            is_rate = (pi_log_probs.detach() - mu_log_probs).exp()
            c = is_rate.clamp_max(c_hat)
            rho = is_rate.clamp_max(rho_hat)

            # Optimistic reward
            rewards_ = rewards.exp() - 1.0

            ###### V-trace / IMPALA
            # https://arxiv.org/abs/1802.01561
            v, advantages = compute_vtrace(values, rewards_, c, rho, self.gamma)

            value_loss = 0.5 * (v - values).pow(2).sum()
            policy_loss = -(pi_log_probs * advantages).sum()
            entropy_loss = -entropy.sum()
            ######

            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.update_state_dict()

            print("v_loss {:.3f} p_loss {:.3f} entropy_loss {:.5f} loss {:.3f}".format(value_loss.item(), policy_loss.item(), entropy_loss.item(), loss.item()))

            #if (i % self.args.save_interval == 0) and not self.args.dummy:
            #    torch.save(self.shared_state_dict.state_dict(), self.args.result_dir / "model.pth")
            #    torch.save(self.shared_state_dict.state_dict(), self.args.result_dir / '..' / 'latest' / "model.pth")

            #if batch_iter == 1:
            #    t_ = time.perf_counter()
            #    i += 1
            #    n_steps = (observations.shape[0] - 1) * observations.shape[1] * i
            #    print("Iteration: {} / Time: {:.3f}s / Total frames {} / Value loss {:.3f} / Policy loss {:.3f} / Entropy loss {:.5f} / Total loss {:.3f} / Reward: {:.3f}".format(
            #        i,
            #        t_ - t,
            #        n_steps,
            #        value_loss.item() / rho.shape[0],
            #        policy_loss.item() / rho.shape[0],
            #        entropy_loss.item() / rho.shape[0],
            #        loss.item() / rho.shape[0],
            #        rewards_.mean().item() * 3600 / self.args.act_every,
            #    ))
            #    t = t_

            # Prevent from replaying the batch if the experience is too much off-policy
            if is_rate.log2().mean().abs() > 0.015:
                batch_iter = self.max_batch_repeat

            time.sleep(0.1)

# V-trace
# dsV = ps ( rs + y V(xs1) - V(xs))
# vs = V(xs) + dsV + y cs(vs1 - V(xs1))
# https://arxiv.org/abs/1802.01561
def compute_vtrace(values, rewards, c, rho, discounts):
    # print("values, rewards, c, rho")
    # print(values.shape, rewards.shape, c.shape, rho.shape)
    with torch.no_grad():
        v = [values[-1]]
        for s in reversed(range(values.shape[0]-1)):
            dV = rho[s] * (rewards[s] + discounts * values[s+1] - values[s])
            v.append(values[s] + dV + discounts * c[s] * (v[-1] - values[s+1]))

        v = torch.stack(tuple(reversed(v)), dim=0)
        advantages = rho * (rewards + discounts * v[1:] - values[:-1])
        return v.detach(), advantages.detach()