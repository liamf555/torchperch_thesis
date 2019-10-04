import gym
import gym_bixler
import bixler
import time

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
args = parser.parse_args()

def process_msg(msg):
    # Process incoming message
    # Convert state into required format for agent
    return np.array([[msg.x,     msg.y,        msg.z,
                      msg.phi,   msg.theta,    msg.psi,
                      msg.u,     msg.v,        msg.w,
                      msg.p,     msg.q,        msg.r,
	                  ]])

def check_heartbeat(master):
    if time.time() - check_heartbeat.last_heartbeat_time >= 1.0:
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
        check_heartbeat.last_heartbeat_time = time.time()
        print('Sent heartbeat')

# Setup the model for inference
ModelType = args.algorithm
model = ModelType.load(args.trained_model_file.name)

# Establish connection to autopilot
#MAVLINK20=1 python3 -i -c "from pymavlink import mavutil; mav = mavutil.mavlink_connection('/dev/ttyS0,115200')"
master = mavutil.mavlink_connection('/dev/ttyS0', baud=115200, source_system=1, source_component=158)
#master = mavutil.mavlink_connection('/dev/ttyTHS2', baud=57600, source_system=1, source_component=158)

# Wait for ArduPilot to be up and running
master.wait_heartbeat()

# Set up loop for sending heartbeat
check_heartbeat.last_heartbeat_time = time.time() - 10.0
check_heartbeat(master)


while True:
    # Send heartbeat if needed
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
            master.mav.param_value_send("",0,master.mavlink.MAV_PARAM_TYPE_UINT8,0,0)
        continue

    # Extract state from message	
    real_state = process_msg(msg)

    # Normalise state for model
    # TODO: env may be None, unclear if it is stored when saved
    obs = model.get_env().normalise_state(real_state)
    bixler.set_state(state)
    q_matrix = model( torch.from_numpy(bixler.get_normalized_state()).double() )

    # Get optimal? action
    action, _states = model.predict(obs, deterministic=True)

    # Pass action on to autopilot
    real_action = model.get_env().controller.convert_action(action)
    master.mav.mlagent_action_send(1,1,real_action[0,0],real_action[0,1])
