import torch, sys, argparse, numpy as np

import matplotlib.pyplot as plt

import controllers
import scenarios

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

parser = argparse.ArgumentParser(description='Q-Learning inference for UAV manoeuvres in PyTorch')
parser.add_argument('--controller', type=check_controller, default='sweep_elevator', help='Which controller to use')
parser.add_argument('--scenario', type=check_scenario, default='perching', help='Which scenario to execute in')
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

#scenario_opts = scenario.parser.parse_args(['--no-random-start','--height-limit','5'])

networkFile = args.networkFile

from network import QNetwork
model = QNetwork(scenario.state_dims,scenario.actions)
model.load_state_dict(torch.load(networkFile))

bixler.reset_scenario()

state_history = []

while not bixler.is_terminal():

    bixler_state = np.concatenate( (bixler.get_state(), np.array(bixler.throttle,ndmin=2) ), axis=0 )
    print( ','.join(map(str,bixler_state[:,0])) )
    state_history.append(bixler_state)

    q_matrix = model( torch.from_numpy(bixler.get_normalized_state()).double() )
    max_action = q_matrix.data.max(1,keepdim=False)[1]
    
    bixler.set_action(max_action.item())
    
    bixler.step(0.1)

states = np.array(state_history)

times = np.linspace(0, len(states)*0.1, len(states))

fig, axs = plt.subplots(3,2)

for ax in axs.flat:
    ax.grid()
    ax.set_xlabel('Time (s)')

# Plot theta
axs[0,0].plot(times,np.rad2deg(states[:,4,:]))
axs[0,0].set_ylabel('Pitch angle (deg)')

# Plot q
axs[0,1].plot(times,np.rad2deg(states[:,10,:]))
axs[0,1].set_ylabel('Pitch rate (deg/s)')

# Plot sweep
axs[1,0].plot(times,states[:,12,:])
axs[1,0].set_ylabel('Sweep angle (deg)')

if  controller.__module__.split('.')[-1] == 'sweep_throttle':
    # Plot throttle
    axs[1,1].plot(times,states[:,15,:])
    axs[1,1].set_ylabel('Throttle setting (N)')
else:
    # Plot elev
    axs[1,1].plot(times,states[:,13,:])
    axs[1,1].set_ylabel('Elevator angle (deg)')

# Plot x vs z
axs[2,0].plot(states[:,0,:],-states[:,2,:])
axs[2,0].set_xlabel('x (m)')
axs[2,0].set_ylabel('z (m)')

# Plot airspeed
axs[2,1].plot(times,states[:,6,:])
axs[2,1].set_ylabel('Airspeed (m/s)')

plt.show()
