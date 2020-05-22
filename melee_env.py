from itertools import product
from copy import deepcopy

import numpy as np

import seeding
from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState


max_action = 0x017E
num_melee_actions = 1 + max_action
num_stages = 32
num_characters = 32


NONE_stick = [
    (0.5, 0.5), # Middle
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
    (0.35, 0.5), # Walk left
    (0.65, 0.5), # Walk right
]
A_stick = [
    (0.5, 0.5), # Neutral
    (0.5, 0.0), # Down smash
    (0.5, 1.0), # Up smash
    (0.0, 0.5), # Left smash
    (1.0, 0.5), # Right smash
    (0.35, 0.5), # Left tilt
    (0.65, 0.5), # Right tilt
    (0.5, 0.35), # Down tilt
    (0.5, 0.65), # Up tilt
]
B_stick = [
    (0.5, 0.5), # Neutral
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (0.0, 0.5), # Left
    (1.0, 0.5), # Right
]
Z_stick = [
    (0.5, 0.5), # Neutral
]
Y_stick = [
    (0.5, 0.5), # Neutral
    (0.0, 0.5), # Left
    (1.0, 0.5), # Right
]
L_stick = [
    (0.5, 0.5), # Neutral
    (0.5, 1.0), # Up
    (0.5, 0.0), # Down
    (0.075, 0.25), # Wavedash left full
    (0.925, 0.25), # Wavedash right full
]

_controller = []
for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
    _controller += [SimpleController(*args) for args in product([SimpleButton(button)], stick)]
_controller_states = [a.real_controller for a in _controller]

class MeleeObservationSpace():
    def __init__(self, n):
        self.n = n

class MeleeActionSpace():
    def __init__(self, n):
        assert n >= 0
        self.dtype = np.int64
        self.n = n
        self.np_random = None
        self.seed()

    def sample(self):
        return self.np_random.randint(self.n)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def to_jsonable(self, sample_n):
        return sample_n

    def from_jsonable(self, sample_n):
        return sample_n

    def __contains__(self, x):
        return self.contains(x)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, MeleeActionSpace) and self.n == other.n


class MeleeEnv():
    num_actions = 30
    observation_size = 22

    def __init__(self, act_every=2, episode_length=600, **dolphin_options):
        super(MeleeEnv, self).__init__()
        self.dolphin = DolphinAPI(**dolphin_options)
        self.act_every = act_every
        self.episode_length = episode_length
        self.action_space = MeleeActionSpace(self.num_actions)
        self.observation_space = MeleeObservationSpace(self.observation_size)
        self._has_reset_once = False
        self._previous_dolphin_state = None
        self._dolphin_state = None
        self._steps_complete = 0

    def reset(self):
        if not self._has_reset_once:
            self._has_reset_once = True
            self._previous_dolphin_state = None
            self._dolphin_state = self.dolphin.reset()
        return self._dolphin_state_to_numpy(self._dolphin_state)

    def close(self):
        self.dolphin.close()

    def step(self, action):
        for _ in range(self.act_every - 1):
            self.dolphin.step([_controller_states[action]])

        self._dolphin_state = self.dolphin.step([_controller_states[action]])

        observation = self._dolphin_state_to_numpy(self._dolphin_state)
        reward = self._compute_reward()
        done = self._steps_complete % (self.episode_length - 1) == 0

        self._previous_dolphin_state = deepcopy(self._dolphin_state)

        self._steps_complete += 1
        return observation, reward, done, {}

    def _player_state_to_numpy(self, state):
        return np.array([
            #state.action_state,
            #state.character,
            state.x / 100.0,
            state.y / 100.0,
            state.action_frame / 30.0,
            state.percent / 100.0,
            state.facing,
            1.0 if state.invulnerable else 0.0,
            state.hitlag_frames_left / 30.0,
            state.hitstun_frames_left / 30.0,
            state.shield_size / 60.0,
            1.0 if state.in_air else 0.0,
            state.jumps_used,
        ])

    def _dolphin_state_to_numpy(self, state):
        player1 = self._player_state_to_numpy(state.players[0])
        player2 = self._player_state_to_numpy(state.players[1])
        return np.concatenate((player1, player2))

    def _compute_reward(self):
        reward = 0.0

        if self._player_just_died(1):
            reward = 1.0

        if self._player_just_died(0):
            reward = -1.0

        return reward

    def _percent_taken_by_player(self, player_index):
        return max(0, self._dolphin_state.players[player_index].percent - self._previous_dolphin_state.players[player_index].percent)

    def _player_is_dying(self, player_state):
        # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
        return player_state.action_state <= 0xA

    def _player_just_died(self, player_index):
        return self._player_is_dying(self._dolphin_state.players[player_index]) and not self._player_is_dying(self._previous_dolphin_state.players[player_index])