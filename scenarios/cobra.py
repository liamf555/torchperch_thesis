import numpy as np
import torch

# Cobra scenario

def wrap_class(BixlerClass):
    class CobraBixler(BixlerClass):
        def __init__(self,noise=0.0):
            super().__init__(noise)
            
            self.episode_history = []
        
        def is_out_of_bounds(self):
            def is_in_range(x,lower,upper):
                return lower < x and x < upper
            # Check x remains sensible
            if not is_in_range(self.position_e[0,0],-50,10):
                return True
            # Check y remains sensible
            if not is_in_range(self.position_e[1,0],-2,2):
                return True
            # Check z remains sensible (i.e. not crashed)
            if not is_in_range(self.position_e[2,0],-10,0):
                return True
            # Check u remains sensible, > 0
            if not is_in_range(self.velocity_b[0,0],0,20):
                return True
            return False

        def get_normalized_state(self, state=None):
            if state is None:
                state=self.get_state()[0:12].T
            pb2 = np.pi/2
            mins = np.array([ -50, -2, -10, -pb2, -pb2, -pb2,  0, -2, -5, -pb2, -pb2, -pb2 ])
            maxs = np.array([  10,  2,   1,  pb2,  pb2,  pb2, 20,  2,  5,  pb2,  pb2,  pb2 ])
            return (state-mins)/(maxs-mins)

        def step(self,steptime):
            # Add state to episode history then step
            self.episode_history.append(self.get_state())
            super().step(steptime)

        def get_reward(self):
            if self.is_terminal():
                if self.is_out_of_bounds():
                    return torch.Tensor([-1])
                # Aiming for same as initial state, translated in x
                target_state = np.array([0,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0], dtype='float64')
                cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0,0])
                # Penalise deviation from target state
                cost = np.dot( (np.squeeze(self.get_state()) - target_state) ** 2, cost_vector ) / 2500
                # Add reward for maximum theta over the episode
                self.episode_history = np.array(self.episode_history)
                max_theta = np.max(self.episode_history[:,4,:])
                return torch.Tensor([ ((1 - cost) * 2) - 1 + max_theta/(np.pi/2) ])
            return torch.Tensor([0])

        def is_terminal(self):
            # Terminal point is reaching x=0 wall
            if self.position_e[0,0] > 0:
                return True
            return self.is_out_of_bounds()
        
        def reset_scenario(self):
            # Clear the current episode history
            self.episode_history = []
            # Put bixler in initial state
            initial_state = np.array([[-20,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0]], dtype="float64")
            self.set_state(initial_state)

    return CobraBixler
