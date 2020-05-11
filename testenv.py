from ssbm_gym.dolphin_api import DolphinAPI
from ssbm_gym.ssbm import actionTypes
from copy import deepcopy
import random

maxAction = 0x017E
numActions = 1 + maxAction

numCharacters = 32
numStages = 32
maxJumps = 8

def isDying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player.action_state <= 0xA

def oneHot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y

class EmbedPlayer():
    def __init__(self, flat=True):
        self.flat = flat

    def __repr__(self):
        s = ''
        s += 'character:\tOneHot(' + str(numCharacters) + ')\n'
        s += 'action_state:\tOneHot(' + str(numActions) + ')\n'
        s += 'state:\tList\n'
        s += '\tpercent:\tFloat\n'
        s += '\tfacing:\tFloat\n'
        s += '\tx:\tFloat\n'
        s += '\ty:\tFloat\n'
        s += '\taction_frame:\tFloat\n'
        s += '\tinvulnerable:\tBool\n'
        s += '\thitlag_frames_left:\tFloat\n'
        s += '\thitstun_frames_left:\tFloat\n'
        s += '\tshield_size:\tFloat\n'
        s += '\tin_air:\tBool\n'
        s += '\tjumps_used:\tInt\n'

        return s

    def __call__(self, player_state):
        percent = player_state.percent/100.0
        facing = player_state.facing
        x = player_state.x/10.0
        y = player_state.y/10.0
        action_state = oneHot(player_state.action_state, numActions)
        action_frame = player_state.action_frame/50.0
        character = oneHot(player_state.character, numCharacters)
        invulnerable = 1.0 if player_state.invulnerable else 0
        hitlag_frames_left = player_state.hitlag_frames_left/10.0
        hitstun_frames_left = player_state.hitstun_frames_left/10.0
        jumps_used = int(player_state.jumps_used)
        #charging_smash = 1.0 if player_state.charging_smash else 0.0
        shield_size = player_state.shield_size/100.0
        in_air = 1.0 if player_state.in_air else 0.0

        data = {
            'character': character,
            'action_state': action_state,
            'state': [
                percent,
                facing,
                x, y,
                action_frame,
                invulnerable,
                hitlag_frames_left,
                hitstun_frames_left,
                shield_size,
                in_air,
                jumps_used
            ]
        }

        if self.flat:
            return list(data.values())
        else:
            return data

class EmbedGame():
    def __init__(self, flat=True):
        self.flat = flat
        self.embed_player = EmbedPlayer(self.flat)

    def __repr__(self):
        s = ''
        s += 'player0/player1:\tEmbedPlayer\n'
        s += str(self.embed_player).replace('\n', '\n\t')
        s += 'stage:\tOneHot(' + str(numStages) + ')\n'

        return s

    def __call__(self, game_state):
        player0 = self.embed_player(game_state.players[0])
        player1 = self.embed_player(game_state.players[1])
        stage = oneHot(game_state.stage, numStages)

        data = {
            'player0': player0,
            'player1': player1,
            'stage': stage,
        }

        if self.flat:
            return list(data.values())
        else:
            return data

class DiagonalActionSpace():
    def __init__(self):
        self.actions = [a[0].real_controller for a in actionTypes['diagonal'].actions]
        self.n = len(self.actions)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def sample(self):
        return random.choice(self.actions)

    def from_index(self, n):
        return self.actions[n]

class TestEnv():
    def __init__(self, frame_limit=100000, pid=0, options={}):
        self.api = DolphinAPI(**options)
        self.frame_limit = frame_limit
        self.pid = pid  # player id
        self.obs = None
        self.prev_obs = None
        self.action_space = DiagonalActionSpace()
        self._observation_space = None
        self._embed_obs = EmbedGame(flat=True)

    def is_terminal(self):
        return self.obs.frame >= self.frame_limit

    def reset(self):
        self.obs = self.api.reset()
        return self.embed_obs(self.obs)

    def close(self):
        self.api.close()

    @property
    def observation_space(self):
        if self._observation_space is not None:
            return self._observation_space
        else:
            self._observation_space = str(self.embed_obs)
            return self._embed_obs

    def embed_obs(self, obs):
        return self._embed_obs(obs)

    def act(self, action):
        return self.action_space.from_index(action)

    def compute_reward(self):
        r = 0.0

        if self.prev_obs is not None:
            # Punish dying.
            if not isDying(self.prev_obs.players[self.pid]) and isDying(self.obs.players[self.pid]):
                r -= 1.0

            # Punish taking percent.
            r -= 0.01 * max(0, self.obs.players[self.pid].percent - self.prev_obs.players[self.pid].percent)

            # Reward killing the opponent.
            if not isDying(self.prev_obs.players[1-self.pid]) and isDying(self.obs.players[1-self.pid]):
                r += 1.0

            # Reward putting percent on the opponent.
            r += 0.01 * max(0, self.obs.players[1-self.pid].percent - self.prev_obs.players[1-self.pid].percent)

        return r

    def step(self, action):
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)

        self.obs = self.api.step([self.action_space.from_index(action)])
        reward = self.compute_reward()
        done = self.is_terminal()
        infos = dict({'frame': self.obs.frame})

        return self.embed_obs(self.obs), reward, done, infos