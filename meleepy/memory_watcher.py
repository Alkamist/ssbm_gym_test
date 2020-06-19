import socket
import binascii
from pathlib import Path

import zmq


def chunk(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def parse_message(message):
    lines = message.splitlines()

    assert(len(lines) % 2 == 0)

    diffs = chunk(lines, 2)

    for diff in diffs:
        diff[1] = binascii.unhexlify(diff[1].zfill(8))

    return diffs


class MemoryWatcher:
    def __init__(self, dolphin_user_directory, unique_id=None):
        self.directory = Path(dolphin_user_directory).joinpath("MemoryWatcher")
        if not self.directory.is_dir():
            self.directory.mkdir()

        self.file_path = self.directory.joinpath("MemoryWatcher")

        context = zmq.Context()
        self.socket = context.socket(zmq.REP)

        if unique_id is not None:
            tcp_port = 5555 + unique_id

            with open(self.file_path, 'w') as f:
                f.write(str(tcp_port))

            self.socket.bind("tcp://127.0.0.1:%d" % tcp_port)
        else:
            self.socket.bind("ipc://" + str(self.file_path))

        self.messages = None

    def __del__(self):
        self.socket.close()

    def get_messages(self):
        if self.messages is None:
            message = self.socket.recv()
            message = message.decode('utf-8')
            self.messages = parse_message(message)

        return self.messages

    def advance(self):
        self.socket.send(b'')
        self.messages = None
