import random

def is_dying(player):
    # see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
    return player.action_state <= 0xA

def one_hot(x, n):
    y = n * [0.0]
    y[x] = 1.0
    return y

max_action = 0x017E
num_actions = 1 + max_action
num_characters = 32
num_stages = 32
max_jumps = 8

def embed_player(player_state):
    percent = player_state.percent / 100.0
    facing = player_state.facing
    x = player_state.x / 10.0
    y = player_state.y / 10.0
    action_state = one_hot(player_state.action_state, num_actions)
    action_frame = player_state.action_frame / 50.0
    character = one_hot(player_state.character, num_characters)
    invulnerable = 1.0 if player_state.invulnerable else 0
    hitlag_frames_left = player_state.hitlag_frames_left / 10.0
    hitstun_frames_left = player_state.hitstun_frames_left / 10.0
    jumps_used = int(player_state.jumps_used)
    shield_size = player_state.shield_size / 100.0
    in_air = 1.0 if player_state.in_air else 0.0
    #charging_smash = 1.0 if player_state.charging_smash else 0.0
    return [
        character,
        action_state,
        percent,
        facing,
        x,
        y,
        action_frame,
        invulnerable,
        hitlag_frames_left,
        hitstun_frames_left,
        shield_size,
        in_air,
        jumps_used
    ]

def embed_game(game_state):
    player0 = embed_player(game_state.players[0])
    player1 = embed_player(game_state.players[1])
    stage = one_hot(game_state.stage, num_stages)
    return [
        player0,
        player1,
        stage,
    ]

class ObservationSpace():
    def __init__(self):
        self.data = {}
        self.n = 0

    def sample(self):
        return random.choice(self.data)

    def from_index(self, n):
        return self.data[n]

    def update(self, game_state):
        parsed_state = embed_game(game_state)
        print(parsed_state)
        #self.data = []
        #self.data.append(parsed_state["action_state"])
        #self.n = len(self.data)
