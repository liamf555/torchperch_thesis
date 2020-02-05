import numpy as np
import gym
import gym_bixler
import bixler
import time
import os

import stable_baselines

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN

from pymavlink import mavutil

import argparse

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)
	else:
		msg = "Could not find algorithm: {}".format(algorithm_name)
		raise argparse.ArgumentTypeError(msg)

parser = argparse.ArgumentParser(prog='gym_infer', description="Live inference script for generic stable_baselines model")
parser.add_argument('--algorithm', '-a', type=check_algorithm, required=True)
parser.add_argument('trained_model_file', type=argparse.FileType('r'))
parser.add_argument('--env', type=str, default = 'Bixler-v0')
parser.add_argument('--latency', type = float)
parser.add_argument('--noise', type = float)
parser.add_argument('--no_var_start', action = 'store_false', dest = 'var_start', default = True)
args = parser.parse_args()

def process_msg(msg):
    # Process incoming message
    # Convert state into required format for agent
    return np.array([[msg.x,     msg.y,        msg.z,
                      msg.phi,   msg.theta,    msg.psi,
                      msg.u,     msg.v,        msg.w,
                      msg.p,     msg.q,        msg.r,
                      msg.sweep, msg.elevator
	                  ]])

def check_heartbeat(master):
    if time.time() - check_heartbeat.last_heartbeat_time >= 1.0:
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
        check_heartbeat.last_heartbeat_time = time.time()
        print('Sent heartbeat')

def parse_kwargs(filename, args):
    # parse latency
    if args.latency is None:
        if "nolat" in filename:
            latency = 0.0
        else:
            latency = 0.023
    else:
        latency = args.latency
    #parse variable start conditions
    if args.var_start and "var" not in filename:
        var_start = False 
    elif (args.var_start == False):
        var_start = False
    else:
        var_start = args.var_start
    #parse noise
    if args.noise is None:
        digits = [int(s) for s in filename.split('_') if s.isdigit()]
        if len(digits) > 2:
            noise = float(str(digits[0])+'.'+str(digits[1]))
        else:
            noise = float(digits[0])
    else:
        noise = args.noise

    return {'latency': latency, 
          'noise': noise,
          'var_start': var_start 
          }

class Transform():

    def __init__(self):
        self.o_agent_ekf = np.zeros((3,1))
        self.rotation = np.identity(3)
        self.offset_vector = np.array([[40], [0], [2]])

    def setup(self, state):

        position_ekf = np.transpose(state[:,0:3])
        yaw = state[0,5]

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        rot = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
            ])

        offset_vector_rot = np.matmul(rot, self.offset_vector)

        position = np.transpose(state[:,0:3])

        self.o_agent_ekf = position_ekf + offset_vector_rot

        self.rotation = np.transpose(rot)

    def apply(self, state):

        position_ekf = np.transpose(state[:,0:3])

        rel_pos_ekf = position_ekf - self.o_agent_ekf

        pos_agent = np.matmul(self.rotation, rel_pos_ekf)

        pos_agent[1] = 0

        state[:,0:3] = np.transpose(pos_agent)

        return state

# Setup the model for inference

kwargs = parse_kwargs(os.path.basename(args.trained_model_file.name), args)

env = gym.make('Bixler-v0', **kwargs)

ModelType = args.algorithm
model = ModelType.load(args.trained_model_file.name)
                                                                                   
obs = env.reset()

# Establish connection to autopilot
#MAVLINK20=1 python3 -i -c "from pymavlink import mavutil; mav = mavutil.mavlink_connection('/dev/ttyS0,115200')"
master = mavutil.mavlink_connection('/dev/ttyS0', baud=115200, source_system=1, source_component=158)
#master = mavutil.mavlink_connection('/dev/ttyTHS2', baud=57600, source_system=1, source_component=158)

# Wait for ArduPilot to be up and running
master.wait_heartbeat()

# Set up loop for sending heartbeat
check_heartbeat.last_heartbeat_time = time.time() - 10.0
check_heartbeat(master)

reset_flag = False
emit_action = False
done = True

while True:

    #Send heartbeat if needed
    check_heartbeat(master)

    Check for new MLAGENT_STATE message
    msg = master.recv_msg()
    if msg is None:
        continue
    if not hasattr(msg,'name'):
        continue
    if msg.name is not 'MLAGENT_STATE':
        if msg.name is 'PARAM_REQUEST_LIST':
            # If an attempt to get parameters is made, return a PARAM_VALUE message indicating no parameters
            # master.mav.param_value_send("",0,mavutil.mavlink.MAV_PARAM_TYPE_UINT8,0,0)
            pass
        if msg.name == 'STATUSTEXT':
            if 'enabled' in str(msg.text): # Detected expr mode entry
                print('Experiment enabled, resetting transform')
                reset_flag = True
                emit_action = True
                done = False
            if 'disabled' in str(msg.text): # Experiment done
                print('Experiment disabled, disabling output')
                emit_action = False
                done = True
        continue

    # Extract state from message
    real_state = process_msg(msg)

    # calculate obs first time round and each time switch activated

    if reset_flag == True:
        obs = env.reset()
        reset_flag = False
   
    print("r_ekf: {}, r_a: {}".format( str(real_state[:,0:3]), str(transformed_state[:,0:3]) ) )

    if emit_action:
         # Pass action on to autopilot
         if not done:
        # Get optimal action
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # Get rates from bixler model
        sweep_rate = env.bixler.sweep_rate
        elev_rate = env.bixler.elev_rate

        master.mav.mlagent_action_send(1,1,sweep_rate, elev_rate)
