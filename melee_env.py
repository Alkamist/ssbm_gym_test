import random
from itertools import product
from copy import deepcopy

from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState

max_action = 0x017E
num_actions = 1 + max_action
num_stages = 32
num_characters = 32
num_player_floats = 13
num_players = 2

def is_dying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player.action_state <= 0xA

def one_hot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y

class ActionSpace():
    def __init__(self):
        controller = []
        for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
            controller += [SimpleController(*args) for args in product([SimpleButton(button)], stick)]
        self.data = [a.real_controller for a in controller]
        self.n = len(self.data)

    def sample(self):
        return random.choice(self.data)

    def from_index(self, n):
        return self.data[n]

class ObservationSpace():
    def __init__(self):
        self.data = [0.0] * (num_stages + num_characters + num_actions + (num_player_floats * num_players))
        self.n = len(self.data)

    def sample(self):
        return random.choice(self.data)

    def from_index(self, n):
        return self.data[n]

class MeleeEnv():
    def __init__(self, frame_limit=100000, **kwargs):
        super(MeleeEnv, self).__init__()
        self.dolphin = DolphinAPI(**kwargs)
        self.ai_port = 0
        self.opponent_port = 1
        self.frame_limit = frame_limit
        self._game_state = None
        self._previous_game_state = None

        # Construct the action space. (possible stick/button combinations
        # that the agent can do on a given frame)
        self.action_space = ActionSpace()

        # Construct the observation space. (states that the agent is aware of)
        self.observation_space = ObservationSpace()

    def _get_player_space(self, player_index):
        player_space = []
        player_space.append(one_hot(self._game_state.players[player_index].character, num_characters))
        player_space.append(one_hot(self._game_state.players[player_index].action_state, num_actions))
        player_space.append(self._game_state.players[player_index].action_frame / 10.0)
        player_space.append(self._game_state.players[player_index].x / 10.0)
        player_space.append(self._game_state.players[player_index].y / 10.0)
        player_space.append(self._game_state.players[player_index].percent / 100.0)
        player_space.append(self._game_state.players[player_index].facing)
        player_space.append(self._game_state.players[player_index].invulnerable)
        player_space.append(self._game_state.players[player_index].hitlag_frames_left / 10.0)
        player_space.append(self._game_state.players[player_index].hitstun_frames_left / 10.0)
        player_space.append(self._game_state.players[player_index].shield_size / 100.0)
        player_space.append(1.0 if self._game_state.players[player_index].in_air else 0.0)
        player_space.append(self._game_state.players[player_index].jumps_used)
        if self._previous_game_state is not None:
            player_space.append((self._game_state.players[player_index].x - self._previous_game_state.players[player_index].x) / 10.0)
            player_space.append((self._game_state.players[player_index].y - self._previous_game_state.players[player_index].y) / 10.0)
        else:
            player_space.append(0.0)
            player_space.append(0.0)
        return player_space

    def _percent_taken(self, player_index):
        return max(0, self._game_state.players[player_index].percent - self._previous_game_state.players[player_index].percent)

    def _just_died(self, player_index):
        return is_dying(self._game_state.players[player_index]) and not is_dying(self._previous_game_state.players[player_index])

    def _compute_reward(self):
        r = 0.0

        if self._previous_game_state is not None:
            # Punish dying.
            if self._just_died(self.ai_port):
                r -= 1.0

            # Punish taking percent.
            r -= 0.01 * self._percent_taken(self.ai_port)

            # Reward killing the opponent.
            if self._just_died(self.opponent_port):
                r += 1.0

            # Reward putting percent on the opponent.
            r += 0.01 * self._percent_taken(self.opponent_port)

        return r

    def _update_observation_space(self):
        self.observation_space.data = []
        self.observation_space.data.append(one_hot(self._game_state.stage, num_stages))
        self.observation_space.data.append(self._get_player_space(0))
        self.observation_space.data.append(self._get_player_space(1))

    def reset(self):
        self._previous_game_state = None
        self._game_state = self.dolphin.reset()
        self._update_observation_space()
        return self.observation_space

    def close(self):
        self.dolphin.close()

    def step(self, action):
        if self._game_state is not None:
            self._previous_game_state = deepcopy(self._game_state)

        self._game_state = self.dolphin.step([action])
        self._update_observation_space()
        reward = self._compute_reward()
        done = self._game_state.frame >= self.frame_limit

        return self.observation_space, reward, done, {}

NONE_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.0, 0.5),
    (.35, 0.5),
    (.65, 0.5),
    (1.0, 0.5)
]
A_stick = [
    (0.5, 0.0),
    (0.0, 0.5),
    (.35, 0.5),
    (0.5, 0.5),
    (.65, 0.5),
    (1.0, 0.5),
    (0.5, .35),
    (0.5, .65),
    (0.5, 1.0)
]
B_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.0, 0.5),
    (1.0, 0.5)
]
Z_stick = [
    (0.5, 0.5)
]
Y_stick = [
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5)
]
L_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (.075, 0.25),
    (.925, 0.25)
]