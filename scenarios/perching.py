import numpy as np
import torch

# Perching scenario

options = {
    'random_starts': True
    }

def normalize_state(state):
    pb2 = np.pi/2
    mins = np.array([ -50, -2, -10, -pb2, -pb2, -pb2,  0, -2, -5, -pb2, -pb2, -pb2 ])
    maxs = np.array([  10,  2,   1,  pb2,  pb2,  pb2, 20,  2,  5,  pb2,  pb2,  pb2 ])
    return (state-mins)/(maxs-mins)

def wrap_class(BixlerClass, random_starts=True):
    class PerchingBixler(BixlerClass):
                def __init__(self,noise=0.0):
                    super().__init__(noise)
                
                def is_out_of_bounds(self):
                    def is_in_range(x,lower,upper):
                        return lower < x and x < upper
                    # Prevent flipping over
                    if self.orientation_e[1,0] > np.pi / 2:
                        return True
                    # Check x remains sensible
                    if not is_in_range(self.position_e[0,0],-50,10):
                        return True
                    # Check y remains sensible
                    if not is_in_range(self.position_e[1,0],-2,2):
                        return True
                    # Check z remains sensible (i.e. not crashed)
                    if not is_in_range(self.position_e[2,0],-10,1):
                        return True
                    # Check u remains sensible, > 0
                    if not is_in_range(self.velocity_b[0,0],0,20):
                        return True
                    return False
        
                def get_reward():
                    reward = torch.Tensor([0])
                    if bixlerNaN:
                        reward = torch.Tensor([ -1 ])
                        sys.exit()
                    elif bixler.is_terminal():
                        if bixler.is_out_of_bounds():
                            reward = torch.Tensor([ -1 ])
                        else:
                            cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0,0])
                            cost = np.dot( np.squeeze(bixler.get_state()) ** 2, cost_vector ) / 2500
                            reward = torch.Tensor( ((1 - cost) * 2) - 1 + max_theta/(np.pi/2))
        
                def is_terminal(self):
                    # Terminal point is floor
                    if self.position_e[2,0] > 0:
                        return True
                    return self.is_out_of_bounds()
                
                def reset_scenario(self):
                    initial_state = np.array([[-40,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0]], dtype="float64")
                    if random_starts:
                        # Add noise in x,z to the starting position
                        start_shift = np.array([[ np.random.rand(), 0, np.random.rand() ]])
                        # Scale for +- 1m in each
                        start_shift = (start_shift - 0.5) * 1
                        initial_state += np.concatenate((
                            start_shift,
                            np.zeros((1,12))
                            ), axis=1)
                    self.set_state(initial_state)

    return PerchingBixler
