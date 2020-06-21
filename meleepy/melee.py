import subprocess
import platform
from pathlib import Path

from .setup_dolphin_user import setup_dolphin_user


IS_USING_WINDOWS = platform.system() == "Windows"


class Melee:
    def __init__(self,
                 dolphin_path=None,
                 melee_iso_path=None,
                 player_stats=["human", "ai"],
                 render=True,
                 speed=0,
                 fullscreen=False,
                 audio=False):

        self.current_directory = Path.cwd()

        self.melee_iso_path = str(self.current_directory.joinpath("MeleeISO", "SSBM.iso")) \
                              if melee_iso_path is None else Path(melee_iso_path)

        dolphin_program_name = "Dolphin.exe" if IS_USING_WINDOWS else "dolphin-emu-nogui"
        self.dolphin_path = str(self.current_directory.joinpath("DolphinEmulator", dolphin_program_name)) \
                            if dolphin_path is None else Path(dolphin_path)

        self.user_directory = setup_dolphin_user(
            in_directory=self.current_directory,
            player_stats=["human", "ai"],
            render=render,
            speed=speed,
            fullscreen=fullscreen,
            audio=audio,
        )

        self.render = render

        self.process = None

    def reset(self):
        self.close()
        self._start_process()

    #def step(self, controllers):
    #    return self.state

    def close(self):
        if self.process is not None:
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
