import random

from ssbm_gym.dolphin_api import DolphinAPI
from action_spaces import ActionSpace
from observation_spaces import ObservationSpace

options = dict(
    windows=True,
    render=True,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

dolphin = DolphinAPI(**options)
game_state = dolphin.reset()

action_space = ActionSpace()
observation_space = ObservationSpace()

while True:
    game_state = dolphin.step([action_space.sample()])
    observation_space.update(game_state)
    print(observation_space.n)