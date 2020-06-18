import random


class ActionSelector():
    def __init__(self, num_actions, use_repeating, max_repeats):
        self.num_actions = num_actions
        self.use_repeating = use_repeating
        self.max_repeats = max_repeats
        self.action_to_repeat = 0
        self.action_repeat_count = 0

    def select_action(self, epsilon):
        action = 0
        was_random = False

        if self.use_repeating:
            if self.action_repeat_count > 0:
                action = self.action_to_repeat
                self.action_repeat_count -= 1
                was_random = True

            elif random.random() <= epsilon:
                action = random.randrange(self.num_actions)
                self.action_to_repeat = action
                self.action_repeat_count = random.randrange(self.max_repeats)
                was_random = True

        elif random.random() <= epsilon:
            action = random.randrange(self.num_actions)
            was_random = True

        return action, was_random
