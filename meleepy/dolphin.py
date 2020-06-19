import subprocess
import platform
from pathlib import Path

from .config_templates import *
from .memory_watcher import MemoryWatcher


IS_USING_WINDOWS = platform.system() == "Windows"


CHARACTER_IDS = {
  "falcon": 0x0,
  "dk": 0x1,
  "fox": 0x2,
  "gaw": 0x3,
  "kirby": 0x4,
  "bowser": 0x5,
  "link": 0x6,
  "luigi": 0x7,
  "mario": 0x8,
  "marth": 0x9,
  "mewtwo": 0xA,
  "ness": 0xB,
  "peach": 0xC,
  "pikachu": 0xD,
  "ics": 0xE,
  "puff": 0xF,
  "samus": 0x10,
  "yoshi": 0x11,
  "zelda": 0x12,
  "sheik": 0x13,
  "falco": 0x14,
  "ylink": 0x15,
  "doc": 0x16,
  "roy": 0x17,
  "pichu": 0x18,
  "ganon": 0x19,
}

STAGE_IDS = {
  "fod": 0x2,
  "stadium": 0x3,
  "PeachsCastle": 0x4,
  "KongoJungle": 0x5,
  "Brinstar": 0x6,
  "Corneria": 0x7,
  "yoshis_story": 0x8,
  "Onett": 0x9,
  "MuteCity": 0xA,
  "RainbowCruise": 0xB,
  "jungle_japes": 0xC,
  "GreatBay": 0xD,
  "HyruleTemple": 0xE,
  "BrinstarDepths": 0xF,
  "YoshiIsland": 0x10,
  "GreenGreens": 0x11,
  "Fourside": 0x12,
  "MushroomKingdomI": 0x13,
  "MushroomKingdomII": 0x14,
  "Akaneia": 0x15,
  "Venom": 0x16,
  "PokeFloats": 0x17,
  "BigBlue": 0x18,
  "IcicleMountain": 0x19,
  "IceTop": 0x1A,
  "FlatZone": 0x1B,
  "dream_land": 0x1C,
  "yoshis_island_64": 0x1D,
  "KongoJungle64": 0x1E,
  "battlefield": 0x1F,
  "final_destination": 0x20,
}


class Dolphin:
    def __init__(self,
                 unique_id=0,
                 dolphin_path=None,
                 melee_iso_path=None,

                 render=True,
                 speed=0,
                 fullscreen=False,
                 audio=False,

                 stage="battlefield",

                 player1="ai",
                 char1="falcon",
                 cpu1=9,

                 player2="ai",
                 char2="falcon",
                 cpu2=9):

        self.current_directory = Path.cwd()

        self.user_directory = self.current_directory.joinpath("DolphinUser")

        self.melee_iso_path = str(self.current_directory.joinpath("MeleeISO", "SSBM.iso")) \
                              if melee_iso_path is None else Path(melee_iso_path)

        dolphin_program_name = "Dolphin.exe" if IS_USING_WINDOWS else "dolphin-emu-nogui"
        self.dolphin_path = str(self.current_directory.joinpath("DolphinEmulator", dolphin_program_name)) \
                            if dolphin_path is None else Path(dolphin_path)

        self.unique_id = unique_id

        self.stage = stage

        self.player1 = player1
        self.char1 = char1
        self.cpu1 = cpu1

        self.player2 = player2
        self.char2 = char2
        self.cpu2 = cpu2

        self.render = render
        self.gfx = "OGL" if render else "Null"
        self.audio = "Pulse" if audio else "No audio backend"
        self.speed = speed
        self.fullscreen = fullscreen

        self._create_directories()
        self._create_ai_pipe_config()
        self._create_dolphin_config()
        self._create_melee_config()
        self._create_memory_watcher()

        self.process = None

    def reset(self):
        if self.process is not None:
            self.close()

        #self.state = ssbm.GameMemory()

        self._start_process()

        #self.pads = self.get_pads()

        #self.start_time = time.time()
        #self.update_state()

        #while self.state.players[0].action_state != 322 or self.state.players[1].action_state != 322:
        #    self.memory_watcher.advance()
        #    self.update_state()

        #return self.state

    def close(self):
        if self.process != None:
            self.process.terminate()

        self.process = None

    def _start_process(self):
        process_args = [self.dolphin_path,
                        "--exec", self.melee_iso_path,
                        "--user", self.user_directory]

        if IS_USING_WINDOWS:
            process_args += ["--batch"]
        else:
            if not self.render:
                process_args += ["--platform", "headless"]
            else:
                process_args += ["--platform", "x11"]

        self.process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return self.process

    def _create_directories(self):
        self.config_directory = self.user_directory.joinpath("Config")
        if not self.config_directory.is_dir():
            self.config_directory.mkdir()

        self.game_settings_directory = self.user_directory.joinpath("GameSettings")
        if not self.game_settings_directory.is_dir():
            self.game_settings_directory.mkdir()

        self.pipes_directory = self.user_directory.joinpath("Pipes")
        if not self.pipes_directory.is_dir():
            self.pipes_directory.mkdir()

    def _create_ai_pipe_config(self):
        with open(self.config_directory.joinpath("GCPadNew.ini"), "w") as f:
            player_ids = [i for i, e in enumerate([self.player1, self.player2]) if e == "ai"]

            config = ""

            for player_id in player_ids:
                player_number = player_id + 1
                config += "[GCPad%d]\n" % player_number
                config += "Device = Pipe/p%d\n" % player_number
                config += PIPE_CONFIG

            f.write(config)

    def _create_dolphin_config(self):
        with open(self.config_directory.joinpath("Dolphin.ini"), "w") as f:
            f.write(DOLPHIN_INI.format(
                user=self.user_directory,
                gfx=self.gfx,
                audio=self.audio,
                speed=self.speed,
                fullscreen=self.fullscreen,
                port1 = 12 if self.player1 == "human" else 6,
                port2 = 12 if self.player2 == "human" else 6,
            ))

    def _create_melee_config(self):
        def byte_str(x):
            return "{0:02X}".format(x)

        match_setup_code = BOOT_TO_MATCH.format(
            stage=byte_str(STAGE_IDS[self.stage]),

            player1=byte_str(1 if self.player1 == "cpu" else 0),
            char1=byte_str(CHARACTER_IDS[self.char1]),
            cpu1=byte_str(self.cpu1),

            player2=byte_str(1 if self.player2 == "cpu" else 0),
            char2=byte_str(CHARACTER_IDS[self.char2]),
            cpu2=byte_str(self.cpu2),
        )

        speed_hack_code = ""
        if self.speed != 1:
            speed_hack_code = "$Speed Hack"
            if self.gfx != "Null":
                speed_hack_code += " Render"

        with open(self.game_settings_directory.joinpath("GALE01.ini"), "w") as f:
            f.write(GALE01_INI.format(
                match_setup=match_setup_code,
                speed_hack=speed_hack_code,
            ))

    def _create_memory_watcher(self):
        if IS_USING_WINDOWS:
            self.memory_watcher = MemoryWatcher(dolphin_user_directory=str(self.user_directory), unique_id=self.unique_id)
        else:
            self.memory_watcher = MemoryWatcher(dolphin_user_directory=str(self.user_directory))

    def _create_ai_pads(self):
        ai_pad_ids = []
        if self.player1 == "ai":
            ai_pad_ids.append(0)
        if self.player2 == "ai":
            ai_pad_ids.append(1)

        pipe_paths = [self.pipes_directory.joinpath("p%d" % i) for i in ai_pad_ids]

        self.ai_pads =