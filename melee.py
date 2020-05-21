from itertools import product
from copy import deepcopy

import torch

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

def _wrap_dolphin_player_state(dolphin_state, previous_dolphin_state):
    return dict(
        x = dolphin_state.x,
        y = dolphin_state.y,
        x_velocity = 0.0 if previous_dolphin_state is None else dolphin_state.x - previous_dolphin_state.x,
        y_velocity = 0.0 if previous_dolphin_state is None else dolphin_state.y - previous_dolphin_state.y,
        action_state = dolphin_state.action_state,
        action_frame = dolphin_state.action_frame,
        percent = dolphin_state.percent,
        facing = dolphin_state.facing,
        invulnerable = dolphin_state.invulnerable,
        hitlag_frames_left = dolphin_state.hitlag_frames_left,
        hitstun_frames_left = dolphin_state.hitstun_frames_left,
        shield_size = dolphin_state.shield_size,
        in_air = dolphin_state.in_air,
        jumps_used = dolphin_state.jumps_used,
    )

def _wrap_dolphin_state(dolphin_state, previous_dolphin_state):
    if previous_dolphin_state is not None:
        return {
            "players": [
                _wrap_dolphin_player_state(dolphin_state.players[0], previous_dolphin_state.players[0]),
                _wrap_dolphin_player_state(dolphin_state.players[1], previous_dolphin_state.players[1])
            ]
        }
    else:
        return {
            "players": [
                _wrap_dolphin_player_state(dolphin_state.players[0], None),
                _wrap_dolphin_player_state(dolphin_state.players[1], None)
            ]
        }

def player_state_to_tensor(state):
    player = torch.tensor([
        state["x"] / 100.0,
        state["y"] / 100.0,
        state["x_velocity"] / 100.0,
        state["y_velocity"] / 100.0,
        state["action_frame"] / 30.0,
        state["percent"] / 100.0,
        state["facing"],
        1.0 if state["invulnerable"] else 0.0,
        state["hitlag_frames_left"] / 30.0,
        state["hitstun_frames_left"] / 30.0,
        state["shield_size"] / 60.0,
        1.0 if state["in_air"] else 0.0,
        state["jumps_used"],
        *_one_hot(state["action_state"], num_melee_actions),
        #*_one_hot(state["character"], num_characters),
    ], dtype=torch.float32)
    return player

def melee_state_to_tensor(state):
    return torch.cat((player_state_to_tensor(state["players"][0]), player_state_to_tensor(state["players"][1]))).unsqueeze(0)

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
        self.state_size = 792
        self.num_actions = 30
        self.has_reset = False
        self._dolphin_state = None
        self._previous_dolphin_state = None
        self.step_count_without_reset = 0

    def reset(self):
        if not self.has_reset:
            self._previous_dolphin_state = None
            self._dolphin_state = self.dolphin.reset()
            self.has_reset = True
        return _wrap_dolphin_state(self._dolphin_state, self._previous_dolphin_state)

    def close(self):
        self.dolphin.close()

    def step(self, action):
        self._dolphin_state = self.dolphin.step([_controller_states[action]])
        state = _wrap_dolphin_state(self._dolphin_state, self._previous_dolphin_state)
        if self._dolphin_state is not None:
            self._previous_dolphin_state = deepcopy(self._dolphin_state)
        #if self.step_count_without_reset >= 72000:
        #    self.has_reset = False
        #    self.step_count_without_reset = 0
        #self.step_count_without_reset += 1
        return state