from itertools import product
from copy import deepcopy

import gym

from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState

max_action = 0x017E
num_actions = 1 + max_action
num_stages = 32
num_characters = 32
max_action_frame = 1000
max_hitlag_frames = 4000
max_hitstun_frames = 4000
max_shield_size = 1000.0
max_percent = 1000.0
max_position_offset = 20000.0

def create_player_space():
    return gym.spaces.Dict({
        "character": gym.spaces.Discrete(num_characters),
        "action_state": gym.spaces.Discrete(num_actions),
        "action_frame": gym.spaces.Discrete(max_action_frame),
        "position": gym.spaces.Box(low=-max_position_offset,high=max_position_offset,shape=(2,)),
        "velocity": gym.spaces.Box(low=-max_position_offset,high=max_position_offset,shape=(2,)),
        "percent": gym.spaces.Box(low=0.0,high=max_percent,shape=(1,)),
        "facing": gym.spaces.Discrete(2),
        "invulnerable": gym.spaces.Discrete(2),
        "hitlag_frames_left": gym.spaces.Discrete(max_hitlag_frames),
        "hitstun_frames_left": gym.spaces.Discrete(max_hitstun_frames),
        "shield_size": gym.spaces.Box(low=0.0,high=max_shield_size,shape=(1,)),
        "in_air": gym.spaces.Discrete(2),
        "jumps_used": gym.spaces.Discrete(8),
    })

class MeleeEnv(gym.Env):
    def __init__(self, frame_limit=100000, **kwargs):
        super(MeleeEnv, self).__init__()
        self.dolphin = DolphinAPI(**kwargs)
        self.frame_limit = frame_limit
        self._game_state = None
        self._previous_game_state = None

        # Construct the action space. (possible stick/button combinations
        # that the agent can do on a given frame)
        controller = []
        for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
            controller += [SimpleController(*args) for args in product([SimpleButton(button)], stick)]
        self._actions = [a.real_controller for a in controller]
        self.action_space = gym.spaces.Discrete(len(self._actions))

        # Construct the observation space. (states that the agent is aware of)
        self.observation_space = gym.spaces.Dict({
            "stage": gym.spaces.Discrete(num_stages),
            "player0": create_player_space(),
            "player1": create_player_space(),
        })

    def _compute_reward(self):
        r = 0.0

        #if self.prev_obs is not None:
        #    # Punish dying.
        #    if not isDying(self.prev_obs.players[self.pid]) and isDying(self.obs.players[self.pid]):
        #        r -= 1.0

        #    # Punish taking percent.
        #    r -= 0.01 * max(0, self.obs.players[self.pid].percent - self.prev_obs.players[self.pid].percent)

        #    # Reward killing the opponent.
        #    if not isDying(self.prev_obs.players[1-self.pid]) and isDying(self.obs.players[1-self.pid]):
        #        r += 1.0

        #    # Reward putting percent on the opponent.
        #    r += 0.01 * max(0, self.obs.players[1-self.pid].percent - self.prev_obs.players[1-self.pid].percent)

        return r

    def reset(self):
        self._game_state = self.dolphin.reset()
        return self._game_state

    def close(self):
        self.dolphin.close()

    def step(self, action):
        if self._game_state is not None:
            self._previous_game_state = deepcopy(self._game_state)

        self._game_state = self.dolphin.step([self._actions[action]])

        self.observation_space.stage = self._game_state.stage
        self.observation_space.player0.action_state = self._game_state.players[0].action_state

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