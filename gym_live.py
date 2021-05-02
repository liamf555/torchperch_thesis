# Run with:
# MAVLINK20=1 PYTHONPATH=~/ardupilot/modules/mavlink python3 agent.py
# Pre-reqs:
#  sudo apt install libxslt-dev libxml2
#  python3 -m pip install lxml

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

import gym
import gym_bixler
import pprint

from pathlib import Path
import stable_baselines
import numpy as np
import json
import time
from pymavlink import mavutil
import argparse

from gym_bixler.envs.live_frame_stack import LiveFrameStack

from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)
	else:
		msg = "Could not find algorithm: {}".format(algorithm_name)
		raise argparse.ArgumentTypeError(msg)


parser = argparse.ArgumentParser(prog='gym_infer', description="Live inference script for generic stable_baselines model")
parser.add_argument("param_file", type=Path)
parser.add_argument('model_dir', type=Path)
args = parser.parse_args()

def process_msg(msg):
    # Process incoming message
    # Convert state into required format for agent
    return (
        np.array([[msg.x,     msg.y,        msg.z,
                   msg.phi,   msg.theta,    msg.psi,
                   msg.u,     msg.v,        msg.w,
                   msg.p,     msg.q,        msg.r,
                   msg.sweep, msg.elevator, msg.va
	               ]]),
        msg.tip == 1.0
        )

def check_heartbeat(master):
    if time.time() - check_heartbeat.last_heartbeat_time >= 1.0:
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
        check_heartbeat.last_heartbeat_time = time.time()
        print('Sent heartbeat')


class Transform():

    def __init__(self, start_position):
        self.o_agent_ekf = np.zeros((3,1))
        self.rotation = np.identity(3)
        self.offset_vector = np.array([[-start_position[0]], [0], [-start_position[1]]])

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

with open(args.model_dir / "sim_params.json") as json_file:
    original_params = json.load(json_file)

with open(args.param_file) as live_json_file:
    live_params = json.load(live_json_file)

params = {key: live_params.get(key, original_params[key]) for key in original_params}

framestack_flag =  params.get("framestack")
ModelType = check_algorithm(params.get("algorithm"))
env0 = DummyVecEnv([lambda: gym.make(params.get("env"), parameters=params)])
if framestack_flag:
    env0 = LiveFrameStack(env0, 4)
env = VecNormalize.load((args.model_dir / "vec_normalize"), env0)

start_position = params.get("start_config")

env.training = False
model = ModelType.load(args.model_dir / (str(ModelType.__name__)))
model.set_env(env)


# Establish connection to autopilot
#MAVLINK20=1 python3 -i -c "from pymavlink import mavutil; mav = mavutil.mavlink_connection('/dev/ttyS0,115200')"
master = mavutil.mavlink_connection('/dev/ttyS0', baud=115200, source_system=1, source_component=158)
#master = mavutil.mavlink_connection('/dev/ttyTHS2', baud=57600, source_system=1, source_component=158)
# master = mavutil.mavlink_connection('tcp:127.0.0.1:5763', baud=115200, source_system=1, source_component=158)

# Wait for ArduPilot to be up and running
master.wait_heartbeat()

# Set up loop for sending heartbeat
check_heartbeat.last_heartbeat_time = time.time() - 10.0
check_heartbeat(master)

transform = Transform(start_position)
in_episode = False

pprint.pprint(params)

while True:
    #Send heartbeat if needed
    check_heartbeat(master)

    # Check for new MLAGENT_STATE message
    msg = master.recv_msg()
    if msg is None:
        continue
    if not hasattr(msg,'name'):
        continue
    if msg.name is not 'MLAGENT_STATE':
        if msg.name is 'PARAM_REQUEST_LIST':
            # If an attempt to get parameters is made, return a PARAM_VALUE message indicating no parameters
            print(mavutil.mavlink.MAV_PARAM_TYPE_UINT8)
            master.mav.param_value_send("",0,mavutil.mavlink.MAV_PARAM_TYPE_UINT8,0,0)
        continue


    # Extract state from message
    (real_state,experimental_mode_enabled) = process_msg(msg)

    if (in_episode is True) and (experimental_mode_enabled is False):
        print("Episode abort")
        in_episode = False

    # Check for episode start
    if (in_episode is False) and (experimental_mode_enabled is True):
        print("Episode start")
        transform.setup(real_state)
        in_episode = True
        env.stack_reset()

    transformed_state = transform.apply(real_state.copy())

    if (transformed_state[:,2] > 0) and (in_episode is True):
        master.mav.mlagent_action_send(1,1,float('nan'),float('nan'))
        print("Episode end")
        in_episode = False

    print("r_ekf: {}, r_a: {}".format( str(real_state[:,0:3]), str(transformed_state[:,0:3]) ) )
    if "airspeed" in params.get("scenario"):
        transformed_state = (np.delete(transformed_state, [1, 3, 5, 7, 9, 11], axis=1))
    else:
        transformed_state = (np.delete(transformed_state, [1, 3, 5, 7, 9, 11, 14], axis=1))


	 # Normalise state for model

    obs = env.normalize_obs(transformed_state)
    if framestack_flag:
        obs = env.stack_obs(obs)

	    # # Get optimal action
    action, _states = model.predict(obs, deterministic=True)

    env.env_method("set_bixler_action", (action))

    # Get rates from bixler model
    sweep_rate = env.get_attr("bixler")[0].next_sweep_rate
    elev_rate = env.get_attr("bixler")[0].next_elev_rate

    # print("sweep: {}, elev: {}".format(str(transformed_state[:,6]),str(transformed_state[:,7])))

    if in_episode:
        # Pass action on to autopilot
        master.mav.mlagent_action_send(1,1,sweep_rate, elev_rate)
        print("sweep: {}, elev: {}".format(str(transformed_state[:,6]),str(transformed_state[:,7])))
        print("u: {}, w: {}".format(str(transformed_state[:,3]), str(transformed_state[:,4])))
        print("pitch : {}, pitch_rate: {}".format(str(transformed_state[:,2]),str(transformed_state[:,5])))
        print("airspeed: {}".format(str(real_state[:,14])))
    
 
