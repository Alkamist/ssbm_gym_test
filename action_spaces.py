import random
from itertools import product
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState

class ActionSpace():
    def __init__(self):
        self.controller = self._make_controller()
        self.actions = [a.real_controller for a in self.controller]
        self.n = len(self.controller)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def sample(self):
        return random.choice(self.actions)

    def from_index(self, n):
        return self.actions[n]

    def _make_controller(self):
        controller = []
        for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
            controller += [SimpleController(*args) for args in product([SimpleButton(button)], stick)]
        return controller

NONE_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.0, 0.5),
    (.35, 0.5),
    (.65, 0.5),
    (1.0, 0.5)
]
A_stick = [
    (0.5, 0.0),
    (0.0, 0.5),
    (.35, 0.5),
    (0.5, 0.5),
    (.65, 0.5),
    (1.0, 0.5),
    (0.5, .35),
    (0.5, .65),
    (0.5, 1.0)
]
B_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.0, 0.5),
    (1.0, 0.5)
]
Z_stick = [
    (0.5, 0.5)
]
Y_stick = [
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5)
]
L_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (.075, 0.25),
    (.925, 0.25)
]

#import struct
#class ContinuousActionSpace():
#    def __init__(self):
#        self.controller = RealControllerState()
#        self.n = 7  # mX, mY, A, B, L, Y, Z
#
#    def __repr__(self):
#        return "Continuous(%d)" % self.n
#
#    def sample(self):
#        mX = random.random()
#        mY = random.random()
#        buttons = []
#        if random.random() > 0.75: buttons.append(SimpleButton.A)
#        if random.random() > 0.75: buttons.append(SimpleButton.B)
#        if random.random() > 0.75: buttons.append(SimpleButton.L)
#        if random.random() > 0.75: buttons.append(SimpleButton.Y)
#        if random.random() > 0.75: buttons.append(SimpleButton.Z)
#        controller = RealControllerState()
#        for button in buttons:
#            setattr(controller, "button_%s" % button.name, True)
#
#        controller.stick_MAIN = (mX, mY)
#        return controller
#
#    def decode_input(self, data):
#        stick_x, stick_y, A, B, L, Y, Z = struct.unpack('hh?????', data)
#        buttons = []
#        if A: buttons.append(SimpleButton.A)
#        if B: buttons.append(SimpleButton.B)
#        if L: buttons.append(SimpleButton.L)
#        if Y: buttons.append(SimpleButton.Y)
#        if Z: buttons.append(SimpleButton.Z)
#        return (stick_x / 255, stick_y / 255), buttons
#
#    def get_controller(self, data):
#        controller = RealControllerState()
#        stick, buttons = self.decode_input(data)
#        for button in buttons:
#            setattr(controller, "button_%s" % button.name, True)
#
#        controller.stick_MAIN = stick
#        return controller