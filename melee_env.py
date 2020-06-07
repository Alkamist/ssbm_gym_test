import math
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
    (0.5, 0.5), # Middle
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
    (0.35, 0.5), # Left tilt
    (0.65, 0.5), # Right tilt
    (0.5, 0.35), # Down tilt
    (0.5, 0.65), # Up tilt
]
B_stick = [
    (0.5, 0.5), # Middle
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
]
Z_stick = [
    (0.5, 0.5), # Middle
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
]
Y_stick = [
    (0.5, 0.5), # Middle
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
]
L_stick = [
    (0.5, 0.5), # Middle
    (0.5, 0.0), # Down
    (0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
    (0.075, 0.25), # Wavedash left full
    (0.925, 0.25), # Wavedash right full
]

_controller = []
for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
    _controller += [SimpleController.init(*args) for args in product([SimpleButton(button)], stick)]
_controller_states = [a.real_controller for a in _controller]

class MeleeObservationSpace(object):
    def __init__(self, n):
        self.n = n

class MeleeActionSpace(object):
    def __init__(self, n, seed):
        assert n >= 0
        self.dtype = np.int64
        self.n = n
        self.np_random = None
        self.seed(seed)

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


def one_hot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y


class MeleeEnv(object):
    #num_actions = 31
    num_actions = 38
    #observation_size = 856
    observation_size = 792

    def __init__(self, seed=None, **dolphin_options):
        self.dolphin = DolphinAPI(**dolphin_options)
        self.seed = seed
        self.action_space = MeleeActionSpace(self.num_actions, seed)
        self.observation_space = MeleeObservationSpace(self.observation_size)
        self._previous_dolphin_state = None
        self._dolphin_state = None

    def reset(self):
        self._previous_dolphin_state = None
        self._dolphin_state = self.dolphin.reset()
        return [self._dolphin_state_to_numpy(0), self._dolphin_state_to_numpy(1)]

    def close(self):
        self.dolphin.close()

    def step(self, actions):
        self._dolphin_state = self.dolphin.step([_controller_states[actions[0]], _controller_states[actions[1]]])

        observations = [self._dolphin_state_to_numpy(0), self._dolphin_state_to_numpy(1)]
        rewards = [self._compute_reward(0), self._compute_reward(1)]
        dones = [self._player_just_died(0), self._player_just_died(1)]
        #dones = [False, False]

        self._previous_dolphin_state = deepcopy(self._dolphin_state)

        return observations, rewards, dones, {}

    def _player_state_to_numpy(self, state, previous_state):
        return np.array([
            #*one_hot(state.character, num_characters),
            *one_hot(state.action_state, num_melee_actions),
            state.x / 100.0,
            state.y / 100.0,
            (state.x - previous_state.x) if previous_state is not None else 0.0,
            (state.y - previous_state.y) if previous_state is not None else 0.0,
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

    def _dolphin_state_to_numpy(self, player_perspective):
        state = self._dolphin_state
        previous_state = self._previous_dolphin_state
        if previous_state is not None:
            main_player = self._player_state_to_numpy(state.players[player_perspective], previous_state.players[player_perspective])
            other_player = self._player_state_to_numpy(state.players[1 - player_perspective], previous_state.players[1 - player_perspective])
        else:
            main_player = self._player_state_to_numpy(state.players[player_perspective], None)
            other_player = self._player_state_to_numpy(state.players[1 - player_perspective], None)
        return np.concatenate((main_player, other_player))

#    def _compute_reward(self, player_perspective):
#        target_location = 0.0
#        reward = 1.0 if abs(self._dolphin_state.players[player_perspective].x - target_location) < 5.0 else 0.0
#        return reward

#    def _compute_reward(self, player_perspective):
#        target_location = 0.0
#        reward = max(-1.0, 1.0 - 0.03 * abs(self._dolphin_state.players[player_perspective].x - target_location))
#        return reward

    def _compute_reward(self, player_perspective):
        main_player = player_perspective
        other_player = 1 - player_perspective

        reward = 0.0

        reward += min(1.0, 0.003 * self._percent_taken_by_player(other_player))
        reward -= min(1.0, 0.003 * self._percent_taken_by_player(main_player))

        if self._player_just_died(other_player):
            reward = 1.0

        if self._player_just_died(main_player):
            reward = -1.0

        return reward

#    def _compute_reward(self, player_perspective):
#        main_player = player_perspective
#        other_player = 1 - player_perspective
#
#        reward = 0.0
#
#        if self._player_just_died(other_player):
#            reward = 1.0
#
#        if self._player_just_died(main_player):
#            reward = -1.0
#
#        return reward

#    def _compute_reward(self, player_perspective):
#        main_player = player_perspective
#        other_player = 1 - player_perspective
#
#        reward = 0.0
#
#        reward += 0.001 * self._percent_taken_by_player(other_player)
#        #reward -= 0.01 * self._percent_taken_by_player(main_player)
#
#        main_x = self._dolphin_state.players[main_player].x
#        other_x = self._dolphin_state.players[other_player].x
#        main_y = self._dolphin_state.players[main_player].y
#        other_y = self._dolphin_state.players[other_player].y
#        distance = math.sqrt((main_x - other_x)**2 + (main_y - other_y)**2)
#
#        reward += 0.0005 * (20.0 - distance)
#
#        #if self._player_just_died(other_player):
#        #    reward = 1.0
#
#        if self._player_just_died(main_player):
#            reward = -1.0
#
#        return reward

    def _percent_taken_by_player(self, player_index):
        if self._previous_dolphin_state is None:
            return 0.0
        else:
            return max(0, self._dolphin_state.players[player_index].percent - self._previous_dolphin_state.players[player_index].percent)

    def _player_is_dying(self, player_state):
        # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
        return player_state.action_state <= 0xA

    def _player_just_died(self, player_index):
        if self._previous_dolphin_state is None:
            return self._player_is_dying(self._dolphin_state.players[player_index])
        else:
            return self._player_is_dying(self._dolphin_state.players[player_index]) and not self._player_is_dying(self._previous_dolphin_state.players[player_index])