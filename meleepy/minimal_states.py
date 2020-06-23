from . import enums


class MinimalPlayerState:
    def __init__(self):
        self.character = enums.Character.UNKNOWN_CHARACTER
        self.action = enums.Action.UNKNOWN_ANIMATION
        self.action_frame = 0
        self.x = 0
        self.y = 0
        self.percent = 0.0
        self.stock = 0
        self.is_facing_right = True
        self.is_invulnerable = False
        self.is_in_hitlag = False
        self.hitstun_frames_left = 0
        self.jumps_left = 0
        self.is_on_ground = True
        self.speed_air_x_self = 0
        self.speed_y_self = 0.0
        self.speed_x_attack = 0.0
        self.speed_y_attack = 0.0
        self.speed_ground_x_self = 0.0

        # Menu stats
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
        self.character_selected = enums.Character.UNKNOWN_CHARACTER
        self.coin_down = False


class MinimalProjectileState:
    def __init(self):
        self.x = 0
        self.y = 0
        self.x_speed = 0
        self.y_speed = 0
        self.subtype = enums.ProjectileSubtype.UNKNOWN_PROJECTILE


class MinimalGameState:
    def __init__(self):
        self.event_size = [0] * 0x100
        self.frame = 0
        self.frame_num = 0
        self.stage = enums.Stage.FINAL_DESTINATION
        self.stage_select_cursor_x = 0.0
        self.stage_select_cursor_y = 0.0
        self.menu_state = enums.Menu.IN_GAME
        self.ready_to_start = False
        self.players = dict()
        self.players[1] = MinimalPlayerState()
        self.players[2] = MinimalPlayerState()
        self.players[3] = MinimalPlayerState()
        self.players[4] = MinimalPlayerState()
        self.players[5] = MinimalPlayerState()
        self.players[6] = MinimalPlayerState()
        self.players[7] = MinimalPlayerState()
        self.players[8] = MinimalPlayerState()
        self.projectiles = []
