from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'done'))


class RolloutMemory:
    # TODO: replace sample list with rollout memory
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque()
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
