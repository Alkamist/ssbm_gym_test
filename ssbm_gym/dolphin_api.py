"""
Responsible for interfacing with Dolphin to interface with SSBM, and handles things like:
* character selection
* stage selection
* running Phillip within SSBM

"""

import os
import time
import random
import functools
import atexit
import platform

from . import ssbm, state_manager, util
from .dolphin import DolphinRunner, Player
from . import memory_watcher as mw
from .state import *
from .pad import *
from . import ctype_util as ct
from .default import *

def get_worker_id():
    file_name = "temp_worker_id"
    max_workers = 1024
    worker_id = 0
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            worker_id = int(f.read())
    with open(file_name, 'w') as f:
        f.write(str((worker_id + 1) % (max_workers - 1)))
    return worker_id

def mw_tcp_port_from_worker_id(worker_id):
    return 5555 + worker_id

class DolphinAPI(Default):
    _options = [
        Option('zmq', type=int, default=1, help="use zmq for memory watcher"),
        # Option('start', type=int, default=1, help="start game in endless time mode"),
        # Option('debug', type=int, default=0),
    ]
    _members = [
        ('dolphin', DolphinRunner),
    ]

    def __init__(self, **kwargs):
        self.windows = platform.system() == "Windows"

        Default.__init__(self, windows=self.windows, **kwargs)

        #self.worker_id = kwargs.get("worker_id") or 0

        #self.user = os.path.expanduser(self.user)
        self.user = self.dolphin.user

        self.worker_id = get_worker_id()

        # set up players
        self.pids = []
        self.players = {}
        # self.levels = {}
        # self.characters = {}
        for i in range(2):
            j = i + 1
            player = getattr(self.dolphin, 'player%d' % j)
            self.players[i] = player
            if player == 'ai':
                self.pids.append(i)

        self.state = ssbm.GameMemory()
        # track players 1 and 2 (pids 0 and 1)
        self.sm = state_manager.StateManager([0, 1])
        self.write_locations()

        # print('Creating MemoryWatcher.')
        mw_path = self.user + '/MemoryWatcher/MemoryWatcher'
        if self.windows:
            mw_port = mw_tcp_port_from_worker_id(self.worker_id)
            util.makedirs(self.user + '/MemoryWatcher')
            with open(mw_path, 'w') as f:
                f.write(str(mw_port))
            self.mw = mw.MemoryWatcherZMQ(port=mw_port)
        else:
            mwType = mw.MemoryWatcherZMQ if self.zmq else mw.MemoryWatcher
            self.mw = mwType(path=mw_path)

        self.dolphin_process = None
        self.last_frame = 0

        pipe_dir = os.path.join(self.user, 'Pipes')
        # print('Creating Pads at %s.' % pipe_dir)
        util.makedirs(pipe_dir)

        pad_ids = self.pids
        pipe_paths = [os.path.join(pipe_dir, 'p%d' % i) for i in pad_ids]
        makePad = functools.partial(Pad, tcp=self.windows, worker_id=self.worker_id)
        self.get_pads = util.async_map(makePad, pipe_paths)

    def reset(self):
        if self.dolphin_process is not None:
            self.close()
            #self.dolphin_process.terminate()
        self.state = ssbm.GameMemory()

        # print('Running dolphin.')
        self.dolphin_process = self.dolphin()
        atexit.register(self.dolphin_process.kill)

        self.pads = self.get_pads()

        # print("Pipes initialized.")

        self.start_time = time.time()
        self.update_state()

        while(self.state.players[0].action_state != 322 or
                    self.state.players[1].action_state != 322):
            self.mw.advance()
            self.update_state()
            # print(self.state)
            # print(self.state.players[0].action_state, self.state.players[1].action_state)
        return self.state

    def close(self):
        for pad in self.pads:
            del pad
        #del self.mw
        self.dolphin_process.terminate()
        time.sleep(0.1)
        self.dolphin_process.terminate()
        self.dolphin_process = None

    def write_locations(self):
        path = os.path.join(self.dolphin.user, 'MemoryWatcher')
        util.makedirs(path)
        # print('Writing locations to:', path)
        with open(os.path.join(path, 'Locations.txt'), 'w') as f:
            f.write('\n'.join(self.sm.locations()))

    def update_state(self):
        messages = self.mw.get_messages()
        for message in messages:
            self.sm.handle(self.state, *message)

    def step(self, controllers):
        for pid, pad in zip(self.pids, self.pads):
            assert(self.players[pid] == 'ai')
            pad.send_controller(controllers[pid])
        while self.state.frame == self.last_frame:
            try:
                self.mw.advance()
                self.update_state()
            except:
                pass

        self.last_frame = self.state.frame
        return self.state

