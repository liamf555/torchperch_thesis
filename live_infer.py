import torch, sys, argparse, numpy as np
import time, infer
from pymavlink import mavutil

def process_msg(msg):
    # Process incoming message
    # Convert state into required format for agent
    return np.array([[msg.x,     msg.y,        msg.z,
                      msg.phi,   msg.theta,    msg.psi,
                      msg.u,     msg.v,        msg.w,
                      msg.p,     msg.q,        msg.r,
                      msg.sweep, msg.elevator, msg.tip ]])

def check_heartbeat(master):
    if time.time() - check_heartbeat.last_heartbeat_time >= 1.0:
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,0)
        check_heartbeat.last_heartbeat_time = time.time()
	print('Sent heartbeat')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning inference for UAV manoeuvres in PyTorch')
    parser.add_argument('--no-graph', action='store_false', dest='graph', default=True)
    parser.add_argument('--stdout', action='store_true', dest='stdout', default=False)
    parser.add_argument('--controller', type=infer.check_controller, default='sweep_elevator', help='Which controller to use')
    parser.add_argument('--scenario', type=infer.check_scenario, default='perching', help='Which scenario to execute in')
    parser.add_argument('--scenario-opts', nargs=1, type=str, default='', help='Options for the scenario')
    parser.add_argument('networkFile', type=str, help='The file to obatain weights from')
    args = parser.parse_args()

    controller = args.controller
    scenario = args.scenario

    # Scenario name: scenario.__name__.split('.')[-1]
    # Controller name: controller.__module__.split('.')[-1]

    scenario_args = None
    if len(args.scenario_opts) is not 0:
        scenario_args = scenario.parser.parse_args(args.scenario_opts[0].split(' '))
    else:
        scenario_args = scenario.parser.parse_args([])

    bixler = scenario.wrap_class(controller, scenario_args)()

    networkFile = args.networkFile

    from network import QNetwork
    model = QNetwork(scenario.state_dims,scenario.actions)
    model.load_state_dict(torch.load(networkFile))

    # Establish connection to autopilot
    master = mavutil.mavlink_connection('/dev/ttyTHS2', baud=57600, source_system=1, source_component=158)
    
    # Wait for ArduPilot to be up and running
    master.wait_heartbeat()
    
    # Set up loop for sending heartbeat
    check_heartbeat.last_heartbeat_time = time.time() - 10.0
    check_heartbeat(master)
    
    while True: # Main loop
        check_heartbeat(master) # Send heartbeat if needed
        
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

        state = process_msg(msg)
        bixler.set_state(state)
        q_matrix = model( torch.from_numpy(bixler.get_normalized_state()).double() )
        max_action = q_matrix.data.max(1,keepdim=False)[1]
        # Convert action into an elevator rate
        bixler.set_action(max_action)
        print('Processing MLAGENT_STATE message. Rate {}'.format(bixler.elev_rate))
        master.mav.mlagent_action_send(1,1,bixler.sweep_rate,10)
