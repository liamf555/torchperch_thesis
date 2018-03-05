import random

class ReplayMemory():
    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.Transition = Transition
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        '''Add a transition to the memory'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
