from struct import unpack, error

import .enums
from .slippstream import SlippstreamClient, CommType, EventType


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
        self.players = [MinimalPlayerState() for _ in range(8)]
        self.projectiles = []

    def parse_slippi_event(self, event_bytes):
        """ Handle a series of events, provided sequentially in a byte array """

        self.menu_state = enums.Menu.IN_GAME

        while len(event_bytes) > 0:
            event_size = self.event_size[event_bytes[0]]

            if len(event_bytes) < event_size:
                print("WARNING: Something went wrong unpacking events. Data is probably missing")
                print("\tDidn't have enough data for event")
                return False

            if EventType(event_bytes[0]) == EventType.PAYLOADS:
                cursor = 0x2
                payload_size = event_bytes[1]
                num_commands = (payload_size - 1) // 3
                for i in range(0, num_commands):
                    command, command_len = unpack(">bH", event_bytes[cursor:cursor+3])
                    self.event_size[command] = command_len+1
                    cursor += 3
                event_bytes = event_bytes[payload_size + 1:]

            elif EventType(event_bytes[0]) == EventType.FRAME_START:
                self.frame_num = unpack(">i", event_bytes[1:5])[0]
                event_bytes = event_bytes[event_size:]

            elif EventType(event_bytes[0]) == EventType.GAME_START:
                event_bytes = event_bytes[event_size:]

            elif EventType(event_bytes[0]) == EventType.GAME_END:
                event_bytes = event_bytes[event_size:]

            elif EventType(event_bytes[0]) == EventType.PRE_FRAME:
                event_bytes = event_bytes[event_size:]

            elif EventType(event_bytes[0]) == EventType.POST_FRAME:
                self.frame = unpack(">i", event_bytes[0x1:0x1+4])[0]
                controller_port = unpack(">B", event_bytes[0x5:0x5+1])[0] + 1

                self.players[controller_port].x = unpack(">f", event_bytes[0xa:0xa+4])[0]
                self.players[controller_port].y = unpack(">f", event_bytes[0xe:0xe+4])[0]

                self.players[controller_port].character = enums.Character(unpack(">B", event_bytes[0x7:0x7+1])[0])

                try:
                    self.players[controller_port].action = enums.Action(unpack(">H", event_bytes[0x8:0x8+2])[0])
                except ValueError:
                    self.players[controller_port].action = enums.Action.UNKNOWN_ANIMATION

                # Melee stores this in a float for no good reason. So we have to convert
                facing_float = unpack(">f", event_bytes[0x12:0x12+4])[0]
                self.players[controller_port].is_facing_right = facing_float > 0

                self.players[controller_port].percent = int(unpack(">f", event_bytes[0x16:0x16+4])[0])
                self.players[controller_port].stock = unpack(">B", event_bytes[0x21:0x21+1])[0]
                self.players[controller_port].action_frame = int(unpack(">f", event_bytes[0x22:0x22+4])[0])

                # Extract the bit at mask 0x20
                bitflags2 = unpack(">B", event_bytes[0x27:0x27+1])[0]
                self.players[controller_port].is_in_hitlag = bool(bitflags2 & 0x20)

                try:
                    self.players[controller_port].hitstun_frames_left = int(unpack(">f", event_bytes[0x2b:0x2b+4])[0])
                except ValueError:
                    self.players[controller_port].hitstun_frames_left = 0

                self.players[controller_port].is_on_ground = not bool(unpack(">B", event_bytes[0x2f:0x2f+1])[0])
                self.players[controller_port].jumps_left = unpack(">B", event_bytes[0x32:0x32+1])[0]
                self.players[controller_port].is_invulnerable = int(unpack(">B", event_bytes[0x34:0x34+1])[0]) != 0

                self.players[controller_port].speed_air_x_self = unpack(">f", event_bytes[0x35:0x35+4])[0]
                self.players[controller_port].speed_y_self = unpack(">f", event_bytes[0x39:0x39+4])[0]
                self.players[controller_port].speed_x_attack = unpack(">f", event_bytes[0x3d:0x3d+4])[0]
                self.players[controller_port].speed_y_attack = unpack(">f", event_bytes[0x41:0x41+4])[0]
                self.players[controller_port].speed_ground_x_self = unpack(">f", event_bytes[0x45:0x45+4])[0]

                event_bytes = event_bytes[event_size:]

            elif EventType(event_bytes[0]) == EventType.GECKO_CODES:
                event_bytes = event_bytes[event_size:]

            elif EventType(event_bytes[0]) == EventType.FRAME_BOOKEND:
                event_bytes = event_bytes[event_size:]
                return True

            elif EventType(event_bytes[0]) == EventType.ITEM_UPDATE:
                projectile = MinimalProjectileState()
                projectile.x = unpack(">f", event_bytes[0x14:0x14+4])[0]
                projectile.y = unpack(">f", event_bytes[0x18:0x18+4])[0]
                projectile.x_speed = unpack(">f", event_bytes[0x0c:0x0c+4])[0]
                projectile.y_speed = unpack(">f", event_bytes[0x10:0x10+4])[0]

                try:
                    projectile.subtype = enums.ProjectileSubtype(unpack(">H", event_bytes[0x05:0x05+2])[0])
                except ValueError:
                    projectile.subtype = enums.ProjectileSubtype.UNKNOWN_PROJECTILE

                self.projectiles.append(projectile)

                event_bytes = event_bytes[event_size:]

            else:
                print("WARNING: Something went wrong unpacking events. " + \
                    "Data is probably missing")
                print("\tGot invalid event type: ", event_bytes[0])
                return False

        return False

    def parse_slippi_menu_event(self, event_bytes):
        scene = unpack(">H", event_bytes[0x1:0x1+2])[0]

        if scene == 0x02:
            self.menu_state = enums.Menu.CHARACTER_SELECT
        if scene == 0x0102:
            self.menu_state = enums.Menu.STAGE_SELECT

        # CSS Cursors
        self.players[1].cursor_x = unpack(">f", event_bytes[0x3:0x3+4])[0]
        self.players[1].cursor_y = unpack(">f", event_bytes[0x7:0x7+4])[0]
        self.players[2].cursor_x = unpack(">f", event_bytes[0xB:0xB+4])[0]
        self.players[2].cursor_y = unpack(">f", event_bytes[0xF:0xF+4])[0]
        self.players[3].cursor_x = unpack(">f", event_bytes[0x13:0x13+4])[0]
        self.players[3].cursor_y = unpack(">f", event_bytes[0x17:0x17+4])[0]
        self.players[4].cursor_x = unpack(">f", event_bytes[0x1B:0x1B+4])[0]
        self.players[4].cursor_y = unpack(">f", event_bytes[0x1F:0x1F+4])[0]

        # Ready to fight banner
        self.ready_to_start = unpack(">B", event_bytes[0x23:0x23+1])[0] == 0

        # Stage
        try:
            self.stage = enums.Stage(unpack(">B", event_bytes[0x24:0x24+1])[0])
        except ValueError:
            self.stage = enums.Stage.NO_STAGE

        # controller port statuses at CSS
        try:
            self.players[1].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x25:0x25+1])[0])
        except error:
            self.players[1].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
        try:
            self.players[2].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x26:0x26+1])[0])
        except error:
            self.players[2].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
        try:
            self.players[3].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x27:0x27+1])[0])
        except error:
            self.players[3].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
        try:
            self.players[4].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x28:0x28+1])[0])
        except error:
            self.players[4].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED

        # Character selected
        try:
            id = unpack(">B", event_bytes[0x29:0x29+1])[0]
            self.players[1].character_selected = enums.convertToInternalCharacterID(id)
        except error:
            self.players[1].character_selected = enums.Character.UNKNOWN_CHARACTER
        try:
            id = unpack(">B", event_bytes[0x2A:0x2A+1])[0]
            self.players[2].character_selected = enums.convertToInternalCharacterID(id)
        except error:
            self.players[2].character_selected = enums.Character.UNKNOWN_CHARACTER
        try:
            id = unpack(">B", event_bytes[0x2B:0x2B+1])[0]
            self.players[3].character_selected = enums.convertToInternalCharacterID(id)
        except error:
            self.players[3].character_selected = enums.Character.UNKNOWN_CHARACTER
        try:
            id = unpack(">B", event_bytes[0x2C:0x2C+1])[0]
            self.players[4].character_selected = enums.convertToInternalCharacterID(id)
        except error:
            self.players[4].character_selected = enums.Character.UNKNOWN_CHARACTER

        # Coin down
        try:
            self.players[1].coin_down = unpack(">B", event_bytes[0x2D:0x2D+1])[0] == 2
        except error:
            self.players[1].coin_down = False
        try:
            self.players[2].coin_down = unpack(">B", event_bytes[0x2E:0x2E+1])[0] == 2
        except error:
            self.players[2].coin_down = False
        try:
            self.players[3].coin_down = unpack(">B", event_bytes[0x2F:0x2F+1])[0] == 2
        except error:
            self.players[3].coin_down = False
        try:
            self.players[4].coin_down = unpack(">B", event_bytes[0x30:0x30+1])[0] == 2
        except error:
            self.players[4].coin_down = False

        # Stage Select Cursor X, Y
        self.stage_select_cursor_x = unpack(">f", event_bytes[0x31:0x31+4])[0]
        self.stage_select_cursor_y = unpack(">f", event_bytes[0x35:0x35+4])[0]

        # Frame count
        self.frame = unpack(">i", event_bytes[0x39:0x39+4])[0]
