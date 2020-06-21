import errno
import socket
from struct import pack, unpack, error
from enum import Enum
from hexdump import hexdump
from ubjson.decoder import DecoderException
import ubjson

from . import enums
from .minimal_states import MinimalGameState


# pylint: disable=too-few-public-methods
class EventType(Enum):
    """ Replay event types """
    GECKO_CODES = 0x10
    PAYLOADS = 0x35
    GAME_START = 0x36
    PRE_FRAME = 0x37
    POST_FRAME = 0x38
    GAME_END = 0x39
    FRAME_START = 0x3a
    ITEM_UPDATE = 0x3b
    FRAME_BOOKEND = 0x3c


class CommType(Enum):
    """ Types of SlippiComm messages """
    HANDSHAKE = 0x01
    REPLAY = 0x02
    KEEPALIVE = 0x03
    MENU = 0x04


class SlippiCommClient():
    """ Implementation of a SlippiComm client.

    This can be used to talk to some server implementing the SlippiComm protocol
    (i.e. the Project Slippi fork of Nintendont or Slippi Ishiiruka).
    """

    def __init__(self, address="", port=51441, realtime=True):
        self.buf = bytearray()
        self.server = None
        self.realtime = realtime
        self.address = address
        self.port = port

    def shutdown(self):
        """ Close down the socket and connection to the console. """
        if self.server is not None:
            self.server.close()
            return True
        return False

    def read_message(self):
        """ Read an entire message from the registered socket.

        Returns None on failure, Dict of data from ubjson on success.
        """
        while True:
            try:
                # The first 4 bytes are the message's length
                #   read this first
                while len(self.buf) < 4:
                    self.buf += self.server.recv(4 - len(self.buf))
                    if len(self.buf) == 0:
                        return None
                message_len = unpack(">L", self.buf[0:4])[0]

                # Now read in message_len amount of data
                while len(self.buf) < (message_len + 4):
                    self.buf += self.server.recv((message_len + 4) - len(self.buf))

                try:
                    # Exclude the the message length in the header
                    msg = ubjson.loadb(self.buf[4:])
                    # Clear out the old buffer
                    del self.buf
                    self.buf = bytearray()
                    return msg

                except DecoderException as exception:
                    print("ERROR: Decode failure in SlippiComm")
                    print(exception)
                    print(hexdump(self.buf[4:]))
                    self.buf.clear()
                    return None

            except socket.error as exception:
                if exception.args[0] == errno.EWOULDBLOCK:
                    continue
                print("ERROR with socket:", exception)
                return None

    def connect(self):
        """ Connect to the server.

        Returns True on success, False on failure.
        """
        # If we don't have a slippi address, let's autodiscover it
        if not self.address:
            # Slippi broadcasts a UDP message on port
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Slippi sends an advertisement every 10 seconds. So 20 should be enough
            sock.settimeout(20)
            sock.bind(('', 20582))
            try:
                message = sock.recvfrom(1024)
                self.address = message[1][0]
            except socket.timeout:
                return False

        if self.server is not None:
            return True

        # Try to connect to the server and send a handshake
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.server.connect((self.address, self.port))
            self.server.send(self.__new_handshake())
        except socket.error as exception:
            if exception.args[0] == errno.ECONNREFUSED:
                self.server = None
                return False
            self.server = None
            return False

        return True

    def __new_handshake(self, cursor=None, token=None):
        """ Returns a new binary handshake message. """
        cursor = cursor or [0, 0, 0, 0, 0, 0, 0, 0]
        token = token or [0, 0, 0, 0, 0, 0, 0, 0]

        handshake = bytearray()
        handshake_contents = ubjson.dumpb({
            'type': CommType.HANDSHAKE.value,
            'payload': {
                'cursor': cursor,
                'clientToken': token,
                'isRealtime': self.realtime,
            }
        })
        handshake += pack(">L", len(handshake_contents))
        handshake += handshake_contents
        return handshake


def parse_slippi_event(event_bytes, game_state):
    """ Handle a series of events, provided sequentially in a byte array """

    game_state.menu_state = enums.Menu.IN_GAME

    while len(event_bytes) > 0:
        event_size = game_state.event_size[event_bytes[0]]

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
                game_state.event_size[command] = command_len+1
                cursor += 3
            event_bytes = event_bytes[payload_size + 1:]

        elif EventType(event_bytes[0]) == EventType.FRAME_START:
            game_state.frame_num = unpack(">i", event_bytes[1:5])[0]
            event_bytes = event_bytes[event_size:]

        elif EventType(event_bytes[0]) == EventType.GAME_START:
            event_bytes = event_bytes[event_size:]

        elif EventType(event_bytes[0]) == EventType.GAME_END:
            event_bytes = event_bytes[event_size:]

        elif EventType(event_bytes[0]) == EventType.PRE_FRAME:
            event_bytes = event_bytes[event_size:]

        elif EventType(event_bytes[0]) == EventType.POST_FRAME:
            game_state.frame = unpack(">i", event_bytes[0x1:0x1+4])[0]
            controller_port = unpack(">B", event_bytes[0x5:0x5+1])[0] + 1

            game_state.players[controller_port].x = unpack(">f", event_bytes[0xa:0xa+4])[0]
            game_state.players[controller_port].y = unpack(">f", event_bytes[0xe:0xe+4])[0]

            game_state.players[controller_port].character = enums.Character(unpack(">B", event_bytes[0x7:0x7+1])[0])

            try:
                game_state.players[controller_port].action = enums.Action(unpack(">H", event_bytes[0x8:0x8+2])[0])
            except ValueError:
                game_state.players[controller_port].action = enums.Action.UNKNOWN_ANIMATION

            # Melee stores this in a float for no good reason. So we have to convert
            facing_float = unpack(">f", event_bytes[0x12:0x12+4])[0]
            game_state.players[controller_port].is_facing_right = facing_float > 0

            game_state.players[controller_port].percent = int(unpack(">f", event_bytes[0x16:0x16+4])[0])
            game_state.players[controller_port].stock = unpack(">B", event_bytes[0x21:0x21+1])[0]
            game_state.players[controller_port].action_frame = int(unpack(">f", event_bytes[0x22:0x22+4])[0])

            # Extract the bit at mask 0x20
            bitflags2 = unpack(">B", event_bytes[0x27:0x27+1])[0]
            game_state.players[controller_port].is_in_hitlag = bool(bitflags2 & 0x20)

            try:
                game_state.players[controller_port].hitstun_frames_left = int(unpack(">f", event_bytes[0x2b:0x2b+4])[0])
            except ValueError:
                game_state.players[controller_port].hitstun_frames_left = 0

            game_state.players[controller_port].is_on_ground = not bool(unpack(">B", event_bytes[0x2f:0x2f+1])[0])
            game_state.players[controller_port].jumps_left = unpack(">B", event_bytes[0x32:0x32+1])[0]
            game_state.players[controller_port].is_invulnerable = int(unpack(">B", event_bytes[0x34:0x34+1])[0]) != 0

            game_state.players[controller_port].speed_air_x_self = unpack(">f", event_bytes[0x35:0x35+4])[0]
            game_state.players[controller_port].speed_y_self = unpack(">f", event_bytes[0x39:0x39+4])[0]
            game_state.players[controller_port].speed_x_attack = unpack(">f", event_bytes[0x3d:0x3d+4])[0]
            game_state.players[controller_port].speed_y_attack = unpack(">f", event_bytes[0x41:0x41+4])[0]
            game_state.players[controller_port].speed_ground_x_self = unpack(">f", event_bytes[0x45:0x45+4])[0]

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

            game_state.projectiles.append(projectile)

            event_bytes = event_bytes[event_size:]

        else:
            print("WARNING: Something went wrong unpacking events. " + \
                "Data is probably missing")
            print("\tGot invalid event type: ", event_bytes[0])
            return False

    return False


def parse_slippi_menu_event(event_bytes, game_state):
    scene = unpack(">H", event_bytes[0x1:0x1+2])[0]

    if scene == 0x02:
        game_state.menu_state = enums.Menu.CHARACTER_SELECT
    if scene == 0x0102:
        game_state.menu_state = enums.Menu.STAGE_SELECT

    # CSS Cursors
    game_state.players[1].cursor_x = unpack(">f", event_bytes[0x3:0x3+4])[0]
    game_state.players[1].cursor_y = unpack(">f", event_bytes[0x7:0x7+4])[0]
    game_state.players[2].cursor_x = unpack(">f", event_bytes[0xB:0xB+4])[0]
    game_state.players[2].cursor_y = unpack(">f", event_bytes[0xF:0xF+4])[0]
    game_state.players[3].cursor_x = unpack(">f", event_bytes[0x13:0x13+4])[0]
    game_state.players[3].cursor_y = unpack(">f", event_bytes[0x17:0x17+4])[0]
    game_state.players[4].cursor_x = unpack(">f", event_bytes[0x1B:0x1B+4])[0]
    game_state.players[4].cursor_y = unpack(">f", event_bytes[0x1F:0x1F+4])[0]

    # Ready to fight banner
    game_state.ready_to_start = unpack(">B", event_bytes[0x23:0x23+1])[0] == 0

    # Stage
    try:
        game_state.stage = enums.Stage(unpack(">B", event_bytes[0x24:0x24+1])[0])
    except ValueError:
        game_state.stage = enums.Stage.NO_STAGE

    # controller port statuses at CSS
    try:
        game_state.players[1].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x25:0x25+1])[0])
    except error:
        game_state.players[1].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
    try:
        game_state.players[2].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x26:0x26+1])[0])
    except error:
        game_state.players[2].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
    try:
        game_state.players[3].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x27:0x27+1])[0])
    except error:
        game_state.players[3].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
    try:
        game_state.players[4].controller_status = enums.ControllerStatus(unpack(">B", event_bytes[0x28:0x28+1])[0])
    except error:
        game_state.players[4].controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED

    # Character selected
    try:
        id = unpack(">B", event_bytes[0x29:0x29+1])[0]
        game_state.players[1].character_selected = enums.convertToInternalCharacterID(id)
    except error:
        game_state.players[1].character_selected = enums.Character.UNKNOWN_CHARACTER
    try:
        id = unpack(">B", event_bytes[0x2A:0x2A+1])[0]
        game_state.players[2].character_selected = enums.convertToInternalCharacterID(id)
    except error:
        game_state.players[2].character_selected = enums.Character.UNKNOWN_CHARACTER
    try:
        id = unpack(">B", event_bytes[0x2B:0x2B+1])[0]
        game_state.players[3].character_selected = enums.convertToInternalCharacterID(id)
    except error:
        game_state.players[3].character_selected = enums.Character.UNKNOWN_CHARACTER
    try:
        id = unpack(">B", event_bytes[0x2C:0x2C+1])[0]
        game_state.players[4].character_selected = enums.convertToInternalCharacterID(id)
    except error:
        game_state.players[4].character_selected = enums.Character.UNKNOWN_CHARACTER

    # Coin down
    try:
        game_state.players[1].coin_down = unpack(">B", event_bytes[0x2D:0x2D+1])[0] == 2
    except error:
        game_state.players[1].coin_down = False
    try:
        game_state.players[2].coin_down = unpack(">B", event_bytes[0x2E:0x2E+1])[0] == 2
    except error:
        game_state.players[2].coin_down = False
    try:
        game_state.players[3].coin_down = unpack(">B", event_bytes[0x2F:0x2F+1])[0] == 2
    except error:
        game_state.players[3].coin_down = False
    try:
        game_state.players[4].coin_down = unpack(">B", event_bytes[0x30:0x30+1])[0] == 2
    except error:
        game_state.players[4].coin_down = False

    # Stage Select Cursor X, Y
    game_state.stage_select_cursor_x = unpack(">f", event_bytes[0x31:0x31+4])[0]
    game_state.stage_select_cursor_y = unpack(">f", event_bytes[0x35:0x35+4])[0]

    # Frame count
    game_state.frame = unpack(">i", event_bytes[0x39:0x39+4])[0]


class Slippi():
    def __init__(self, address="", port=51441, realtime=True):
        self.client = SlippiCommClient(address, port, realtime)

    def get_game_state(self):
        game_state = MinimalGameState()

        # Keep looping until we get a REPLAY message
        frame_ended = False
        while not frame_ended:
            msg = self.client.read_message()
            if msg:
                if CommType(msg['type']) == CommType.REPLAY:
                    event = msg['payload']['data']
                    frame_ended = parse_slippi_event(event, game_state)

                # We can basically just ignore keepalives
                elif CommType(msg['type']) == CommType.KEEPALIVE:
                    pass

                elif CommType(msg['type']) == CommType.HANDSHAKE:
                    handshake = msg['payload']
                    print("Connected to console '{}' (Slippi Nintendont {})".format(
                        handshake['nick'],
                        handshake['nintendontVersion'],
                    ))

                # Handle menu-state event
                elif CommType(msg['type']) == CommType.MENU:
                    event = msg['payload']['data']
                    parse_slippi_menu_event(event, game_state)
                    frame_ended = True

        return game_state
