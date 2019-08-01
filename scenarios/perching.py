import numpy as np
#import torch
import argparse

# Perching scenario

parser = argparse.ArgumentParser(prog='Perching Scenario', usage='--scenario-opts "[options]"')
parser.add_argument('--no-random-start', action='store_false', dest='random_start', default=True)
parser.add_argument('--height-limit', type=float, default=10)


state_dims = 14
actions = 49
failReward = -1.0

def wrap_class(BixlerClass, options):
    class PerchingBixler(BixlerClass):
                def __init__(self,noise=0.0):
                    super(PerchingBixler,self).__init__(noise)
                
                def is_out_of_bounds(self):
                    h_min = -options.height_limit
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
                    if not is_in_range(self.position_e[2,0],h_min,1):
                        return True
                    # Check u remains sensible, > 0
                    if not is_in_range(self.velocity_b[0,0],0,20):
                        return True
                    return False
        
                def get_reward(self):
                    if self.is_terminal():
                        if self.is_out_of_bounds():
                            return failReward
                        cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0 ])
                        cost = np.dot( np.squeeze(self.get_state()) ** 2, cost_vector ) / 2500
                        return  ((1.0 - cost) * 2.0) - 1.0
                    return 0.0
        
                def is_terminal(self):
                    # Terminal point is floor
                    if self.position_e[2,0] > 0:
                        return True
                    return self.is_out_of_bounds()

                def get_state(self):
                    return super(PerchingBixler,self).get_state()[0:14].T

                def get_normalized_state(self, state=None):
                    if state is None:
                        state=self.get_state()
                    pb2 = np.pi/2
                    h_min = -options.height_limit
                    mins = np.array([ -50, -2, h_min, -pb2, -pb2, -pb2,  0, -2, -5, -pb2, -pb2, -pb2, self.sweep_limits[0], self.elev_limits[0]])
                    maxs = np.array([  10,  2,     1,  pb2,  pb2,  pb2, 20,  2,  5,  pb2,  pb2,  pb2, self.sweep_limits[1], self.elev_limits[1]])
                    return (state-mins)/(maxs-mins)
                
                def reset_scenario(self):
                    initial_state = np.array([[-40,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0]], dtype="float64")
                    if options.random_start:
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
