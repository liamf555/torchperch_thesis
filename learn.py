import random
import math
from collections import namedtuple
from itertools import count

import warnings

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from network import QNetwork
from bixler import Bixler
from replay import ReplayMemory

initial_state = np.array([[-40,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0]])

model = QNetwork()
bixler = Bixler()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
memory = ReplayMemory(100000,Transition)
optimizer = torch.optim.RMSprop(model.parameters())

EPS_START = 1.0
EPS_END   = 0.1
EPS_DECAY = 1000000 - 10000
BATCH_SIZE = 32
GAMMA = 0.99

sync_rate = 100 # Update target network every 100 iterations

steps = 0

def select_action(state):
    def should_take_max():
        global steps
        sample = random.random()
        epsilon_threshold = 1.0
        if steps < 10000:
            epsilon_threshold = EPS_START
        else:
            epsilon_threshold = EPS_END + (EPS_START - EPS_END) * math.exp( -1.0 * (steps-10000) / EPS_DECAY )
        steps += 1
        if sample > epsilon_threshold:
            return True
        return False
    
    if should_take_max():
        return model(Variable(torch.from_numpy(state).double())).data.max(1)[1].view(1,1)
    else:
        return torch.LongTensor([[random.randrange(49)]])


def train_on_experience():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Split up transitions list into args & pass to zip
    # zip returns an iterator of tuples:
    #  [(state, state, state, ...), (action, action, action, ...), ...]
    # Split up into args & construct Transition with tuples in each slot
    batch = Transition(*zip(*transitions))
    
    # Get byte tensor with True where next_state is not None
    non_final_mask = torch.ByteTensor(
        tuple(map(lambda s: s is not None, batch.next_state))
        )

    with torch.no_grad():
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch)
    
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.DoubleTensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    with torch.no_grad():
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.double()

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values[:,None])

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 500000

for episode_num in range(num_episodes):
    # Initialise bixler state
    bixler.set_state(initial_state)
    
    state = bixler.get_state()[0:12].T
    next_state = bixler.get_state()[0:12].T
    

    bixlerNaN = False

    for t in count():
        action = select_action(state)
        # Apply action to bixler
        bixler.set_action(action[0,0])
        # Update bixler state
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                for i in range(1,10):
                    bixler.step(0.01)
            except Warning as e:
                    # Set NaN state...
                    bixlerNaN = True

        # Check for NaNs in bixler state
        
        # Get a reward for the transition
        reward = torch.Tensor([0])
        if bixlerNaN:
            reward = torch.Tensor([ -1 ])
        elif bixler.is_terminal():
            cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0,0])
            cost = np.dot( np.squeeze(bixler.get_state()) ** 2, cost_vector ) / 2500
            reward = torch.Tensor([ ((1 - cost) * 2) - 1 ])
        
        # Observe the new state
        if bixler.is_terminal():
            next_state = None
        else:
            next_state = bixler.get_state()[0:12].T

        memory.push(
            torch.from_numpy(state).double(),
            action,
            torch.from_numpy(next_state).double() if next_state is not None else None,
            reward
            )
        
        state = next_state
        
        train_on_experience()
        if bixler.is_terminal():
            print('Episode {} completed. {} frames. Reward: {}'.format(episode_num, t, reward[0]))
            break

torch.save(model,'qNetwork.pkl')
