import random
import math
from collections import namedtuple
from itertools import count
import signal

import torch
import torch.nn.functional as F

import numpy as np

from network import QNetwork
from replay import ReplayMemory

import controllers
import scenarios

import argparse, sys, os

def check_controller(controller_name):
    if hasattr(controllers,controller_name):
        return getattr(controllers,controller_name)
    else:
        msg = "Could not find controller {}".format(controller_name)
        raise argparse.ArgumentTypeError(msg)

def check_scenario(scenario_name):
    if hasattr(scenarios,scenario_name):
        return getattr(scenarios,scenario_name)
    else:
        msg = "Could not find scenario {}".format(scenario_name)
        raise argparse.ArgumentTypeError(msg)

def check_folder(folder_name):
    if os.path.exists(folder_name):
        if os.path.isdir(folder_name):
            return folder_name
        else:
            msg = "File {} exists and is not a directory".format(folder_name)
            raise argparse.ArgumentTypeError(msg)
    os.mkdir(folder_name)
    return folder_name

parser = argparse.ArgumentParser(description='Q-Learning for UAV manoeuvres in PyTorch')
parser.add_argument('--controller', type=str, default='sweep_elevator')
parser.add_argument('--scenario', type=check_scenario, default='perching')
parser.add_argument('--scenario-opts', nargs=1, type=str, default='')
parser.add_argument('--logfile', type=argparse.FileType('w'), default='learning_log.txt')
parser.add_argument('--networks', type=check_folder, default='networks' )
parser.add_argument('--no-stdout', action='store_false', dest='use_stdout', default=True)
args = parser.parse_args()

scenario = args.scenario

model = QNetwork(scenario.state_dims,scenario.actions)
#for param in model.parameters():
#    param = random.uniform(-0.1,0.1)

scenario_args = None
if len(args.scenario_opts) is not 0:
    scenario_args = scenario.parser.parse_args(args.scenario_opts[0].split(' '))
else:
    scenario_args = scenario.parser.parse_args([])

bixler = scenario.wrap_class(args.controller, scenario_args)()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
memory = ReplayMemory(100000,Transition)
# RMSprop proved to be unstable
#  Divides gradients by average gradient, if gradients are v. small, results in blow up
#  Provides an epsilon term to improve stability. Default is 1e-8, may need to increase it...
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.0025, eps=1e-2)
#optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025)
#optimizer = torch.optim.Adam(model.parameters(), lr = 0.0025, amsgrad=True)

BATCH_SIZE = 32
GAMMA = 0.99

sync_rate = 100 # Update target network every 100 iterations ( NOT_IMPLEMENTED )

steps = 0

def interp_linear(current,initial,final,start,stop):
    if current < start:
        return initial
    if current > stop:
        return final
    m = (initial - final) / (start - stop)
    c = initial - m*start
    return m*current + c

def get_epsilon(t_global,t_local):
    k_initial = 1
    k_final = 0.05
    k_start = 10000
    k_stop = 100000
    def get_k(t_global):
        return interp_linear(t_global,k_initial,k_final,k_start,k_stop)
    
    j_initial = 1
    j_final = 0
    j_start = 80000
    j_stop = 300000
    def get_j(t_global):
        return interp_linear(t_global,j_initial,j_final,j_start,j_stop)
    
    average_length = 35 # Average length of episode
    x = t_local / average_length
    k = get_k(t_global)
    j = get_j(t_global)
    return (1-k)*(j*x)**2 + k


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
        # Edge case where batch contains only final states stalls here...
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
        param.data -= 0.005 * param.grad.data
    # Apply gradients using the optimizer
    #optimizer.step()
    
    return loss.item()

num_episodes = 5000

max_frames = 500000
total_frames = 0

episode_num = 0

logfile = args.logfile

def on_sigterm(signum,frame):
    logfile.flush()
    logfile.close()
    sys.exit()
signal.signal(signal.SIGTERM, on_sigterm)

#for episode_num in range(num_episodes):
while total_frames < max_frames:
    episode_num = episode_num + 1
    # Initialise bixler state
    bixler.reset_scenario()
    
    # Set up initial state variables
    state = bixler.get_state()
    next_state = bixler.get_state()
    
    bixlerNaN = False

    # Until an episode ends
    for frame_num in count():
        
        # Select an action
        action, q_value = select_action(bixler.get_normalized_state(),total_frames,frame_num)
        # Apply action to bixler
        bixler.set_action(action[0])
        # Update bixler state
        try:
            bixler.step(0.1)
        except FloatingPointError as e:
            # Set NaN state...
            bixlerNaN = True
            if frame_num == 0:
                print("Failed at frame 0. Action: {}, Q: {}".format(action[0],q_value[0]))
                print("Start state: {}".format(state))
                print("Final state: {}".format(bixler.get_state()))
                print(e)
                sys.exit()

        # Check for NaNs in bixler state
        
        # Compute the reward for the new state
        reward = torch.Tensor([scenario.failReward]) # Default reward for NaN state
        if not bixlerNaN:
            reward = bixler.get_reward()
        
        # Observe the new state
        if bixler.is_terminal():
            next_state = None
        else:
            next_state = bixler.get_state()

        # Add the transition to the replay memory
        memory.push(
            torch.from_numpy(bixler.get_normalized_state(state)).double(),
            action,
            torch.from_numpy(bixler.get_normalized_state(next_state)).double() if next_state is not None else None,
            reward
            )
        
        # Update the stored state for next iteration's experience storage
        state = next_state
        
        # Train from the experience bank
        loss = train_on_experience()
        
        # If at end of episode, print some data and break
        if bixler.is_terminal() or bixlerNaN:
            total_frames = total_frames + frame_num
            
            logString = 'T: {:4}({:2}) F: {:7} R: {:8.5f} Q: {:8.5f} E: {:8.5f} L: {:8.5f}'.format(episode_num, frame_num, total_frames, reward[0], q_value[0], get_epsilon(total_frames,0), loss)
            logfile.write(logString + '\n')
            if args.use_stdout:
                print(logString, flush=True)
            
            if (episode_num % 1000) == 0:
                # Save the network every 1000 episodes
                torch.save(model.state_dict(),'{}/qNetwork_EP{}.pkl'.format(args.networks,episode_num))
            break

# At end of training, save the model
torch.save(model.state_dict(),'{}/qNetwork_final.pkl'.format(args.networks))
