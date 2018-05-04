import torch, sys, numpy as np

import matplotlib.pyplot as plt

from bixler import Bixler

initial_state = np.array([[-40,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0]])

model = torch.load(sys.argv[1])
bixler = Bixler()

def normalize_state(state):
    pb2 = np.pi/2
    mins = np.array([ -50, -2, -5, -pb2, -pb2, -pb2,  0, -2, -5, -pb2, -pb2, -pb2 ])
    maxs = np.array([  10,  2,  1,  pb2,  pb2,  pb2, 20,  2,  5,  pb2,  pb2,  pb2 ])
    return (state-mins)/(maxs-mins)

bixler.set_state(initial_state)

state_history = []

while not bixler.is_terminal():

    bixler_state = bixler.get_state()
    print( ','.join(map(str,bixler_state[:,0])) )
    state_history.append(bixler_state)

    q_matrix = model( torch.from_numpy(normalize_state(bixler_state[0:12].T)).double() )
    max_action = q_matrix.data.max(1,keepdim=False)[1]
    
    bixler.set_action(max_action.item())
    
    for i in range(1,10):
        bixler.step(0.01)

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
