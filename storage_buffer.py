import random


class StorageBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.next_index = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add_item(self, item):
        if self.next_index >= len(self.storage):
            self.storage.append(item)
        else:
            self.storage[self.next_index] = item

        self.next_index = (self.next_index + 1) % self.max_size

    def sample_batch(self, batch_size):
        indices = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]

        batch_of_items = []
        for i in indices:
            batch_of_items.append(self.storage[i])

        return batch_of_items
