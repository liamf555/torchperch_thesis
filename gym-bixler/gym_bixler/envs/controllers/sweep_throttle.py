import numpy as np
import controllers.common as common

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_SweepThrottle(common.BixlerController):
    
    def __init__(self,noise=0.0):
        super().__init__(noise)
        
        # Control surface rate / throttle settings per action
        self.sweep_rates = [ -60, -10, -5, 0, 5, 10, 60 ]
        self.throttles = [ 0, 2, 4, 6, 8, 10, 12]
        
        # Control surface rates (internal state)
        self.sweep_rate = 0 # (deg/s)

    def set_action(self,action_index):
        sweep_rate_idx = int(action_index) % 7
        throttle_idx = int(action_index) // 7

        self.sweep_rate = self.sweep_rates[sweep_rate_idx]
        self.throttle = self.throttles[throttle_idx]

    def update_control_surfaces(self,steptime):
        self.sweep = np.clip(self.sweep + self.sweep_rate * steptime, self.sweep_limits[0], self.sweep_limits[1])
        self.elev = 0.0
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
