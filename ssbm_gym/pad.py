import enum
import os

@enum.unique
class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11

@enum.unique
class Trigger(enum.Enum):
    L = 0
    R = 1

@enum.unique
class Stick(enum.Enum):
    MAIN = 0
    C = 1

class Pad:
    pad_id = -1

    @classmethod
    def increment_pad_id(cls):
        cls.pad_id += 1
        return cls.pad_id

    def get_tcp_port(self):
        min_mw_tcp_port = 5555
        max_workers = 11520
        max_pads_per_worker = 4
        start_offset = min_mw_tcp_port + max_workers
        return start_offset + self.pad_id + max_pads_per_worker * self.worker_id

    def __init__(self, path, tcp=False, worker_id=0):
        self.pad_id = self.increment_pad_id()
        self.worker_id = worker_id
        self.tcp = tcp
        if tcp:
            import zmq
            context = zmq.Context()
            port = self.get_tcp_port()

            with open(path, 'w') as f:
                f.write(str(port))

            self.socket = context.socket(zmq.PUSH)
            address = "tcp://127.0.0.1:%d" % port
            print("Binding pad %s to address %s" % (self.pad_id, address))
            self.socket.bind(address)
        else:
            try:
                os.mkfifo(path)
            except FileExistsError:
                pass
            self.pipe = open(path, 'w', buffering=1)

        self.message = ""

    def __del__(self):
        if not self.tcp:
            self.pipe.close()
        else:
            self.socket.close()

    def write(self, command, buffering=False):
        self.message += command + '\n'
        if not buffering:
            self.flush()

    def flush(self):
        if self.tcp:
            self.socket.send_string(self.message)
        else:
            self.pipe.write(self.message)
        self.message = ""

    def press_button(self, button, buffering=False):
        assert button in Button
        self.write('PRESS {}'.format(button.name), buffering)

    def release_button(self, button, buffering=False):
        assert button in Button
        self.write('RELEASE {}'.format(button.name), buffering)

    def press_trigger(self, trigger, amount, buffering=False):
        """Amount is in [0, 1], with 0 as released."""
        assert trigger in Trigger
        self.write('SET {} {:.2f}'.format(trigger.name, amount), buffering)

    def tilt_stick(self, stick, x, y, buffering=False):
        """x and y are in [0, 1], with 0.5 as neutral."""
        assert stick in Stick
        self.write('SET {} {:.2f} {:.2f}'.format(stick.name, x, y), buffering)

    def send_controller(self, controller):
        for button in Button:
            field = 'button_' + button.name
            if hasattr(controller, field):
                if getattr(controller, field):
                    self.press_button(button, True)
                else:
                    self.release_button(button, True)

        # for trigger in Trigger:
        #     field = 'trigger_' + trigger.name
        #     self.press_trigger(trigger, getattr(controller, field))

        for stick in Stick:
            field = 'stick_' + stick.name
            value = getattr(controller, field)
            self.tilt_stick(stick, value.x, value.y, True)

        self.flush()
