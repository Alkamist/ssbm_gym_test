import time
import queue

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from models import Policy

class Learner(object):
    def __init__(self, observation_size, num_actions, lr, discounting, baseline_cost,
                 entropy_cost, grad_norm_clipping, save_interval, seed, episode_steps,
                 shared_state_dict, device):
        self.discounting = discounting
        self.baseline_cost = baseline_cost
        self.entropy_cost = entropy_cost
        self.grad_norm_clipping = grad_norm_clipping
        self.seed = seed
        self.save_interval = save_interval
        self.device = device
        self.shared_state_dict = shared_state_dict
        self.policy = Policy(observation_size, num_actions).to(self.device)
        self.policy.train()
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        self.update_shared_state_dict()
        torch.manual_seed(self.seed)

    def update_shared_state_dict(self):
        self.shared_state_dict.load_state_dict(self.policy.state_dict())

    def learn(self, batch):
        actor_observations, actor_actions, actor_rewards, actor_dones, actor_logits, actor_baselines, actor_rnn_states = self.queue_batch.get(block=True)

        actor_observations = actor_observations.to(self.device)
        actor_actions = actor_actions.to(self.device)
        actor_rewards = actor_rewards.to(self.device)
        actor_dones = actor_dones.to(self.device)
        actor_logits = actor_logits.to(self.device)
        actor_baselines = actor_baselines.to(self.device)
        actor_rnn_states = actor_rnn_states.to(self.device)

        learner_logits, learner_baselines, _, _ = self.policy(actor_observations, actor_rnn_states)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_baselines[-1]

        discounts = (~actor_dones).float() * self.discounting

        vtrace_returns = vtrace_from_logits(
            behavior_policy_logits=actor_logits,
            target_policy_logits=learner_logits,
            actions=actor_actions,
            discounts=discounts,
            rewards=actor_rewards,
            values=learner_baselines,
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_logits,
            actor_actions,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = self.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_baselines
        )
        entropy_loss = self.entropy_cost * compute_entropy_loss(
            learner_logits
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        self.update_shared_state_dict()

        #if (i % self.save_interval == 0):
        torch.save(self.shared_state_dict.state_dict(), "checkpoints/model.pth")

        t_ = time.perf_counter()
        delta_t = t_ - t
        steps = (actor_observations.shape[0] - 1) * actor_observations.shape[1]
        if delta_t > 0.0:
            print("FPS {:.1f} / Total steps {} / Baseline loss {:.3f} / Policy loss {:.3f} / Entropy loss {:.3f} / Total loss {:.3f} / Reward: {:.3f}".format(
                    steps / (t_ - t),
                    steps * i,
                    baseline_loss,
                    pg_loss,
                    entropy_loss,
                    total_loss,
                    actor_rewards.mean().item() * 600.0,
                )
            )
        t = t_

        time.sleep(0.1)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


import collections

VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")

def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)

@torch.no_grad()
def vtrace_from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)

def vtrace_from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = vtrace_from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )