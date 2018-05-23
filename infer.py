import torch, sys, numpy as np

import matplotlib.pyplot as plt

from controllers.sweep_elevator import Bixler_SweepElevator

import scenarios

scenario = None
if hasattr(scenarios,sys.argv[1]):
    scenario = getattr(scenarios, sys.argv[1])
else:
    scenario = scenarios.perching

bixler = scenario.wrap_class(Bixler_SweepElevator,random_starts=False)()

networkFile = sys.argv[1]
if len(sys.argv) > 2:
    networkFile = sys.argv[2]

model = torch.load(networkFile)

bixler.reset_scenario()

state_history = []

while not bixler.is_terminal():

    bixler_state = bixler.get_state()
    print( ','.join(map(str,bixler_state[:,0])) )
    state_history.append(bixler_state)

    q_matrix = model( torch.from_numpy(scenario.normalize_state(bixler_state[0:12].T)).double() )
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
