import torch, sys, numpy as np

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

while not bixler.is_terminal():

    bixler_state = bixler.get_state()
    print(','.join(map(str,bixler_state[:,0])) )

    q_matrix = model( torch.from_numpy(normalize_state(bixler_state[0:12].T)).double() )
    max_action = q_matrix.data.max(1,keepdim=False)[1]
    
    bixler.set_action(max_action.item())
    
    for i in range(1,10):
        bixler.step(0.01)
