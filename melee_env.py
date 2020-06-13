import math
import random
from itertools import product
from copy import deepcopy

import numpy as np

from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState


max_action = 0x017E
num_melee_actions = 1 + max_action
num_stages = 32
num_characters = 32


NONE_stick = [
    #(0.5, 0.5), # Middle
    #(0.5, 0.0), # Down
    #(0.5, 1.0), # Up
    (1.0, 0.5), # Right
    (0.0, 0.5), # Left
    #(0.35, 0.5), # Walk left
    #(0.65, 0.5), # Walk right
]
A_stick = [
    #(0.5, 0.5), # Middle
    #(0.5, 0.0), # Down
    #(0.5, 1.0), # Up
    #(1.0, 0.5), # Right
    #(0.0, 0.5), # Left
    #(0.35, 0.5), # Left tilt
    #(0.65, 0.5), # Right tilt
    #(0.5, 0.35), # Down tilt
    #(0.5, 0.65), # Up tilt
]
B_stick = [
    #(0.5, 0.5), # Middle
    #(0.5, 0.0), # Down
    #(0.5, 1.0), # Up
    #(1.0, 0.5), # Right
    #(0.0, 0.5), # Left
]
Z_stick = [
    #(0.5, 0.5), # Middle
    #(0.5, 0.0), # Down
    #(0.5, 1.0), # Up
    #(1.0, 0.5), # Right
    #(0.0, 0.5), # Left
]
Y_stick = [
    #(0.5, 0.5), # Middle
    #(0.5, 0.0), # Down
    #(0.5, 1.0), # Up
    #(1.0, 0.5), # Right
    #(0.0, 0.5), # Left
]
L_stick = [
    #(0.5, 0.5), # Middle
    #(0.5, 0.0), # Down
    #(0.5, 1.0), # Up
    #(1.0, 0.5), # Right
    #(0.0, 0.5), # Left
    #(0.075, 0.25), # Wavedash left full
    #(0.925, 0.25), # Wavedash right full
]

_controller = []
for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
    _controller += [SimpleController.init(*args) for args in product([SimpleButton(button)], stick)]
_controller_states = [a.real_controller for a in _controller]

class MeleeObservationSpace(object):
    def __init__(self, n):
        self.n = n

class MeleeActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)


def one_hot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y


class MeleeEnv(object):
    #num_actions = 38
    num_actions = 2
    #observation_size = 856
    observation_size = 792

    def __init__(self, **dolphin_options):
        self.dolphin = DolphinAPI(**dolphin_options)
        self.action_space = MeleeActionSpace(self.num_actions)
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

    def _compute_reward(self, player_perspective):
        target_location = 0.0
        reward = max(0.0, 1.0 - 0.03 * abs(self._dolphin_state.players[player_perspective].x - target_location))
        return reward

#    def _compute_reward(self, player_perspective):
#        reward = 0.0
#
#        if self._player_just_died(1 - player_perspective):
#            reward = 1.0
#
#        return reward

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