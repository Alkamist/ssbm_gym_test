import math

import numpy as np
import torch


class MinMaxStats(object):
    """ A class that holds the min-max values of the tree. """

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    @property
    def is_expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, network_output):
        self.hidden_state = network_output.hidden_state
        self.reward = network_output.reward
        # softmax over policy logits
        policy = {a: math.exp(network_output.policy_logits[0][a]) for a in actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTS(object):
    def __init__(self, num_simulations, pb_c_base, pb_c_init, discount):
        self.num_simulations = num_simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount

    def run(self, root, action_history, model):
        min_max_stats = MinMaxStats()

        for _ in range(self.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [node]

            while node.is_expanded:
                action, node = self.select_child(node, min_max_stats)
                history.add_action(action)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            print(parent.hidden_state.device)
            network_output = model.recurrent_inference(parent.hidden_state, torch.tensor([[history.last_action()]], device=parent.hidden_state.device))
            node.expand(history.action_space(), network_output)

            self.backpropagate(search_path, network_output.value.item(), min_max_stats)

    def select_child(self, node, min_max_stats):
        _, action, child = max((self.ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value)

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value)

            value = node.reward + self.discount * value
