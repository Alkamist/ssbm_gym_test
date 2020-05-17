from itertools import product
from copy import deepcopy

from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState

max_action = 0x017E
num_actions = 1 + max_action
num_stages = 32
num_characters = 32

NONE_stick = [
    #(0.5, 0.5),
    #(0.5, 0.0),
    (0.35, 0.5),
    #(.35, 0.5),
    #(.65, 0.5),
    (0.75, 0.5),
    #(0.5, 1.0)
]
A_stick = [
    #(0.5, 0.0),
    #(0.0, 0.5),
    #(.35, 0.5),
    #(0.5, 0.5),
    #(.65, 0.5),
    #(1.0, 0.5),
    #(0.5, .35),
    #(0.5, .65),
    #(0.5, 1.0)
]
B_stick = [
    #(0.5, 0.5),
    #(0.5, 0.0),
    #(0.5, 1.0),
    #(0.0, 0.5),
    #(1.0, 0.5)
]
Z_stick = [
    #(0.5, 0.5)
]
Y_stick = [
    #(0.0, 0.5),
    #(0.5, 0.5),
    #(1.0, 0.5)
]
L_stick = [
    #(0.5, 0.5),
    #(0.5, 0.0),
    #(0.5, 1.0),
    #(.075, 0.25),
    #(.925, 0.25)
]

controller = []
for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
    controller += [SimpleController(*args) for args in product([SimpleButton(button)], stick)]
controller_states = [a.real_controller for a in controller]

def is_dying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player.action_state <= 0xA

def one_hot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y

class Melee():
    def __init__(self, **dolphin_options):
        super(Melee, self).__init__()
        self.dolphin = DolphinAPI(**dolphin_options)
        self.state = None
        self.previous_state = None
        self.num_actions = len(controller_states)

    def embed_player_state(self, player_index):
        state = self.state.players[player_index]
        player = []
        #player += one_hot(state.character, num_characters)
        #player += one_hot(state.action_state, num_actions)
        #player.append(state.action_frame / 30.0)
        player.append(state.x / 100.0)
        player.append(state.y / 100.0)
        #player.append(state.percent / 100.0)
        #player.append(state.facing)
        #player.append(1.0 if state.invulnerable else 0.0)
        #player.append(state.hitlag_frames_left / 30.0)
        #player.append(state.hitstun_frames_left / 30.0)
        #player.append(state.shield_size / 60.0)
        #player.append(1.0 if state.in_air else 0.0)
        #player.append(state.jumps_used)
        #if self.previous_state is not None:
        #    previous_state = self.previous_state.players[player_index]
        #    player.append((state.x - previous_state.x) / 100.0)
        #    player.append((state.y - previous_state.y) / 100.0)
        #else:
        #    player.append(0.0)
        #    player.append(0.0)
        return player

    def embed_state(self):
        return self.embed_player_state(0)
        #return self.embed_player_state(0) + self.embed_player_state(1)

    def percent_taken_by_player(self, player_index):
        return self.state.players[player_index].percent - self.state.players[player_index].percent

    def player_just_died(self, player_index):
        return is_dying(self.state.players[player_index]) and not is_dying(self.previous_state.players[player_index])

    def reset(self):
        self.previous_state = None
        self.state = self.dolphin.reset()
        return self.state

    def close(self):
        self.dolphin.close()

    def step(self, action):
        if self.state is not None:
            self.previous_state = deepcopy(self.state)
        self.state = self.dolphin.step([controller_states[action]])
        return self.state