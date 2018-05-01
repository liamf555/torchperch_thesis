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
#for param in model.parameters():
#    param = random.uniform(-0.1,0.1)
bixler = Bixler()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
memory = ReplayMemory(100000,Transition)
#optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.0025)
#optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025, amsgrad=True)

EPS_START = 1.0
EPS_END   = 0.1
EPS_DECAY = 1000000 - 10000
BATCH_SIZE = 32
GAMMA = 0.99

sync_rate = 100 # Update target network every 100 iterations ( NOT_IMPLEMENTED )

steps = 0

def get_epsilon(t_global,t_local):
    k_initial = 1
    k_final = 0.05
    k_start = 1000
    k_stop = 10000
    def get_k(t_global):
        if t_global < k_start:
            return k_initial
        if t_global > k_stop:
            return k_final
        m = (k_initial - k_final) / (k_start - k_stop)
        c = k_initial - m*k_start
        return m*t_global + c
    
    average_length = 35 # Average length of episode
    x = t_local / average_length
    k = get_k(t_global)
    return (1-k)*x**2 + k

#epsilon_threshold = 1.0
#if steps < 10000:
#    epsilon_threshold = EPS_START
#else:
#    epsilon_threshold = EPS_END + (EPS_START - EPS_END) * math.exp( -1.0 * (steps-10000) / EPS_DECAY )


def select_action(state,t_global,t_local):
    """Select an action based on the epsilon-greedy policy"""
    
    def should_take_max():
        """Return true if random sample > current epsilon"""
        if random.random() > get_epsilon(t_global,t_local):
            return True
        return False
    
    # Get the current Q-values from the network
    with torch.no_grad():
        current_q_matrix = model( torch.from_numpy(state).double() )
    action_index = None
    
    if should_take_max():
        # Get index of the maximum Q value and return it
        action_index = current_q_matrix.data.max(1,keepdim=False)[1]
        #print("Taking max: {}".format(action_index))
    else:
        # Return a random index
        action_index = torch.LongTensor([random.randrange(49)])
        #print("Taking random: {}".format(action_index))
    
    return action_index, current_q_matrix[:,action_index[0]]

def normalize_state(state):
    pb2 = np.pi/2
    mins = np.array([ -50, -2, -5, -pb2, -pb2, -pb2,  0, -2, -5, -pb2, -pb2, -pb2 ])
    maxs = np.array([  10,  2,  1,  pb2,  pb2,  pb2, 20,  2,  5,  pb2,  pb2,  pb2 ])
    return (state-mins)/(maxs-mins)

def train_on_experience():
    """Train the network on experience replay"""
    if len(memory) < BATCH_SIZE:
        # If there aren't enough experiences, skip training
        return np.inf
    
    # Get a sample set of transitions
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
    # Get the set of experiences where next_state is non-terminal

    with torch.no_grad():
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Get the currently predicted Q-values for each of the state-action pairs from the experience batch
    state_action_values = model(state_batch).gather(1, action_batch[:,None])
    
    with torch.no_grad():
        # Set up a tensor of values for the next state
        next_state_values = torch.zeros(BATCH_SIZE).double()
        # For all non-final states, get the max Q-value over all actions
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
        # Calculate the expected value for the state-action pair from the batch
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.double()

    # Effectively corrects against errors for the immediate reward in the batch
    #  i.e. the error in the Q-value predicted for an action-state and the reward from the environment
    # The use of next_state_values makes this prone to instability as large changes in next_state_values
    #  as predicted by the network may mask errors in immediate reward. For bixler case, much of the immediate reward
    #  is zero so should be correcting its future estimates...
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values[:,None])

    #print(loss.item())

    # Optimize the model
    # Zero the gradients held in the model
    optimizer.zero_grad()
    # Propogate the loss backwards to compute the gradients
    loss.backward()
    # For each parameter in the model, restrict the gradient to limit divergence
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
        param.data -= 0.0025 * param.grad.data
    # Apply gradients using the optimizer
    #optimizer.step()
    
    return loss.item()

num_episodes = 5000

max_frames = 50000
total_frames = 0

episode_num = 0

#for episode_num in range(num_episodes):
while total_frames < max_frames:
    episode_num = episode_num + 1
    # Initialise bixler state
    bixler.set_state(initial_state)
    
    # Set up initial state variables
    state = bixler.get_state()[0:12].T
    next_state = bixler.get_state()[0:12].T
    
    # Until an episode ends
    bixlerNaN = False

    for frame_num in count():
        # Select an action
        action, q_value = select_action(normalize_state(state),total_frames,frame_num)
        # Apply action to bixler
        bixler.set_action(action[0])
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
        
        # Compute the reward for the new state
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

        # Add the transition to the replay memory
        memory.push(
            torch.from_numpy(normalize_state(state)).double(),
            action,
            torch.from_numpy(normalize_state(next_state)).double() if next_state is not None else None,
            reward
            )
        
        # Update the stored state for next iteration's experience storage
        state = next_state
        
        # Train from the experience bank
        loss = train_on_experience()
        
        # If at end of episode, print some data and break
        if bixler.is_terminal():
            total_frames = total_frames + frame_num
            
            print('T: {:4}({:2}) F: {:7} R: {:8.5f} Q: {:8.5f} E: {:8.5f} L: {:8.5f}'.format(episode_num, frame_num, total_frames, reward[0], q_value[0], get_epsilon(total_frames,0), loss))
            break

# At end of training, save the model
torch.save(model,'qNetwork.pkl')
