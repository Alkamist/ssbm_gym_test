from itertools import product
from copy import deepcopy

import numpy as np
import torch
from torch import nn

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

def _one_hot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y

def get_player_state(state):
    return np.array([
        state.action_state,
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
        #*_one_hot(state.action_state, num_melee_actions),
        #*_one_hot(state.character, num_characters),
    ])

def melee_state_to_tensor(state, device):
    player1 = get_player_state(state.players[0])
    player2 = get_player_state(state.players[1])
    return torch.tensor([np.concatenate((player1, player2))], device=device, dtype=torch.float32).unsqueeze(0)

#def percent_taken_by_player(self, player_index):
#    return self.state.players[player_index].percent - self.state.players[player_index].percent

def _is_dying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player["action_state"] <= 0xA

def player_just_died(state, next_state, player_index):
    return _is_dying(next_state["players"][player_index]) and not _is_dying(state["players"][player_index])

class Melee():
    def __init__(self, **dolphin_options):
        super(Melee, self).__init__()
        self.dolphin = DolphinAPI(**dolphin_options)
        self.state_size = 24
        self.num_actions = 30

    def reset(self):
        return self.dolphin.reset()

    def close(self):
        self.dolphin.close()

    def step(self, action):
        return self.dolphin.step([_controller_states[action]])