import time
import queue

import torch
from torch import nn
import torch.optim as optim

from models import Policy

class Learner(object):
    def __init__(self, observation_size, num_actions, lr, c_hat, rho_hat, gamma,
                 value_loss_coef, entropy_coef, max_grad_norm, save_interval, seed,
                 queue_batch, shared_state_dict, device):
        self.c_hat = c_hat
        self.rho_hat = rho_hat
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.save_interval = save_interval
        self.device = device
        self.queue_batch = queue_batch
        self.shared_state_dict = shared_state_dict
        self.policy = Policy(observation_size, num_actions).to(self.device)
        self.policy.train()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.update_shared_state_dict()

    def update_shared_state_dict(self):
        self.shared_state_dict.load_state_dict(self.policy.state_dict())

    def learning(self):
        torch.manual_seed(self.seed)

        i = 0
        t = time.perf_counter()
        while True:
            i += 1

            try:
                actor_observations, actor_actions, mu_log_probs, actor_rewards, _ = self.queue_batch.get(block=True)
            except queue.Empty:
                pass

            actor_observations = actor_observations.to(self.device)
            actor_actions = actor_actions.to(self.device)
            mu_log_probs = mu_log_probs.to(self.device)
            actor_rewards = actor_rewards.to(self.device)
            #actor_dones = actor_dones.to(self.device)

            values, pi_log_probs, entropy = self.policy.evaluate_actions(actor_observations, actor_actions)

            is_rate = (pi_log_probs.detach() - mu_log_probs).exp()
            c = is_rate.clamp_max(self.c_hat)
            rho = is_rate.clamp_max(self.rho_hat)

            # Optimistic reward
            rewards_ = actor_rewards.exp() - 1.0

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

            self.update_shared_state_dict()

            if (i % self.save_interval == 0):
                torch.save(self.shared_state_dict.state_dict(), "checkpoints/model" + str(i) + ".pth")

            t_ = time.perf_counter()
            delta_t = t_ - t
            steps = (actor_observations.shape[0] - 1) * actor_observations.shape[1]
            if delta_t > 0.0:
                print("FPS {:.1f} / Total steps {} / Value loss {:.3f} / Policy loss {:.3f} / Entropy loss {:.3f} / Total loss {:.3f} / Reward: {:.3f}".format(
                        steps / (t_ - t),
                        steps * i,
                        value_loss,
                        policy_loss,
                        entropy_loss,
                        loss,
                        actor_rewards.mean().item() * 600.0,
                    )
                )
            t = t_

            time.sleep(0.1)

# V-trace
# dsV = ps(rs + y V(xs1) - V(xs))
# vs = V(xs) + dsV + y cs(vs1 - V(xs1))
# https://arxiv.org/abs/1802.01561
def compute_vtrace(values, rewards, c, rho, discounts):
    with torch.no_grad():
        v = [values[-1]]
        for s in reversed(range(values.shape[0]-1)):
            dV = rho[s] * (rewards[s] + discounts * values[s+1] - values[s])
            v.append(values[s] + dV + discounts * c[s] * (v[-1] - values[s+1]))

        v = torch.stack(tuple(reversed(v)), dim=0)
        advantages = rho * (rewards + discounts * v[1:] - values[:-1])
        return v.detach(), advantages.detach()