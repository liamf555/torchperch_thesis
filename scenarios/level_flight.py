import numpy as np
import torch
import argparse

# Level flight scenario

parser = argparse.ArgumentParser(prog='Level Flight Scenario', usage='--scenario-opts "[options]"')

state_dims = 15
actions = 49
failReward = -10

def wrap_class(BixlerClass,options):
    class LevelFlightBixler(BixlerClass):
        def __init__(self,noise=0.0):
            super().__init__(noise)
        
        def is_out_of_bounds(self):
            def is_in_range(x,lower,upper):
                return lower < x and x < upper
            # Check x remains sensible
            if not is_in_range(self.position_e[0,0],-110,10):
                return True
            # Check y remains sensible
            if not is_in_range(self.position_e[1,0],-2,2):
                return True
            # Check z remains sensible (i.e. not crashed)
            if not is_in_range(self.position_e[2,0],-10,0):
                return True
            # Check u remains sensible, > 0
            if not is_in_range(self.velocity_b[0,0],0,25):
                return True
            return False

        def get_state(self):
            return super().get_state()[0:state_dims].T

        def get_normalized_state(self, state=None):
            if state is None:
                state=self.get_state()
            pb2 = np.pi/2
            mins = np.array([ -50, -2, -10, -pb2, -pb2, -pb2,  0, -2, -5, -pb2, -pb2, -pb2, -10, -10, -10 ])
            maxs = np.array([  10,  2,   1,  pb2,  pb2,  pb2, 25,  2,  5,  pb2,  pb2,  pb2,   30,  10,  10 ])
            return (state-mins)/(maxs-mins)

        def step(self,steptime):
            # Add state to episode history then step
            super().step(steptime)

        def get_reward(self):
            #if self.is_terminal():
            if self.is_out_of_bounds():
                return torch.Tensor([-10])
            # Aiming for same as initial state, translated in x
            target_state = np.array([0,0,-5, 0,0,0, 13,0,0, 0,0,0, 0,0,0], dtype='float64')
            cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0,0])
            # Penalise deviation from target state
            cost = np.dot( (np.squeeze(self.get_state()) - target_state) ** 2, cost_vector ) /2500
            reward = ((1 - cost) * 2) - 1
            return torch.Tensor([ max(reward,failReward) ])
            #return torch.Tensor([0])

        def is_terminal(self):
            # Terminal point is reaching x=0 wall
            if self.position_e[0,0] > 0:
                return True
            return self.is_out_of_bounds()
        
        def reset_scenario(self):
            # Put bixler in initial state
            initial_state = np.array([[-40,0,-5, 0,0,0, 13,0,0, 0,0,0, 0,0,0]], dtype="float64")
            self.set_state(initial_state)

    return LevelFlightBixler
