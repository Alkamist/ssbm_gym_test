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

def is_dying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player.action_state <= 0xA

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
        self.ai_port = 0
        self.opponent_port = 1
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

    def _update_player_in_observation_space(self, player_index):
        player_name = "player" + str(player_index)
        self.observation_space[player_name].character = self._game_state.players[player_index].character
        self.observation_space[player_name].action_state = self._game_state.players[player_index].action_state
        self.observation_space[player_name].action_frame = self._game_state.players[player_index].action_frame
        self.observation_space[player_name].position = [
            self._game_state.players[player_index].x,
            self._game_state.players[player_index].y
        ]
        self.observation_space[player_name].percent = self._game_state.players[player_index].percent
        self.observation_space[player_name].facing = self._game_state.players[player_index].facing
        self.observation_space[player_name].invulnerable = self._game_state.players[player_index].invulnerable
        self.observation_space[player_name].hitlag_frames_left = self._game_state.players[player_index].hitlag_frames_left
        self.observation_space[player_name].hitstun_frames_left = self._game_state.players[player_index].hitstun_frames_left
        self.observation_space[player_name].shield_size = self._game_state.players[player_index].shield_size
        self.observation_space[player_name].in_air = self._game_state.players[player_index].in_air
        self.observation_space[player_name].jumps_used = self._game_state.players[player_index].jumps_used
        if self._previous_game_state is not None:
            self.observation_space[player_name].velocity = [
                self._game_state.players[player_index].x - self._previous_game_state.players[player_index].x,
                self._game_state.players[player_index].y - self._previous_game_state.players[player_index].y
            ]
        else:
            self.observation_space[player_name].velocity = [0.0, 0.0]

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
        self.observation_space.stage = self._game_state.stage
        self._update_player_in_observation_space(0)
        self._update_player_in_observation_space(1)

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

        self._game_state = self.dolphin.step([self._actions[action]])
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