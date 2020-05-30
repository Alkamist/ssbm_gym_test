import math

import numpy as np
import torch

from melee_env import MeleeEnv
from model import MuZeroNet
from mcts import MCTS, Node


melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episode_steps = 600
total_training_steps = 20000


def select_action(node, temperature=1, deterministic=True):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i, _ in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v, _ in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)
    return visit_counts[action_pos][1]


class ActionHistory(object):
    """
    Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history, action_space_size):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action):
        self.history.append(action)

    def last_action(self):
        return self.history[-1]

    def action_space(self):
        return [i for i in range(self.action_space_size)]


def store_search_statistics(action_space_size, child_visits, root_values, root, idx=None):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (i for i in range(action_space_size))
    if idx is None:
        child_visits.append([root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space])
        root_values.append(root.value)
    else:
        child_visits[idx] = [root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space]
        root_values[idx] = root.value


if __name__ == "__main__":
    network = MuZeroNet(
        input_size=MeleeEnv.observation_size,
        num_actions=MeleeEnv.num_actions,
    ).to(device=device)
    network.eval()

    mcts = MCTS(
        num_simulations=3,
        pb_c_base=19652,
        pb_c_init=1.25,
        discount=0.997,
    )

    env = MeleeEnv(**melee_options)

    legal_actions = [a for a in range(env.action_space.n)]
    action_history = ActionHistory([], env.action_space.n)
    child_visits = []
    root_values = []

    observation = env.reset()

    steps_trained = 0

    #for _ in range(episode_steps):
    while True:
        main_player_observation = torch.tensor([observation[0]], dtype=torch.float32, device=device)

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        root.expand(legal_actions, network.initial_inference(main_player_observation))
        root.add_exploration_noise(dirichlet_alpha=0.25, exploration_fraction=0.25)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        mcts.run(root, action_history, network)

#        temperature = 1
#        if steps_trained < 0.5 * total_training_steps:
#            temperature = 1.0
#        elif steps_trained < 0.75 * total_training_steps:
#            temperature = 0.5
#        else:
#            temperature = 0.25
#
#        action = select_action(root, temperature, deterministic=False)

        action = select_action(root, 1, deterministic=True)

        observation, reward, done, _ = env.step([action, 0])

        #store_search_statistics(env.action_space.n, child_visits, root_values, root)
