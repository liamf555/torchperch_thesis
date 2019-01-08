import numpy as np
import controllers.common as common

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_SweepElevator(common.BixlerController):
    
    def __init__(self,noise=0.0):
        super(Bixler_SweepElevator, self).__init__(noise)
        
        # Control surface rates for each action
        self.elev_rates = [-60, -10, -5, 0, 5, 10, 60]
        self.sweep_rates = [ -60, -10, -5, 0, 5, 10, 60 ]
        
        # Control surface rates
        self.sweep_rate = 0 # (deg/s)
        self.elev_rate = 0  # (deg/s)


    def set_action(self,action_index):
        # Action may be control surface rates / throttle
        elev_rate_idx = int(action_index) // 7
        sweep_rate_idx = int(action_index) % 7

        self.elev_rate =  self.elev_rates[elev_rate_idx]
        self.sweep_rate = self.sweep_rates[sweep_rate_idx]
    
    def update_control_surfaces(self,steptime):
        self.sweep = np.clip(self.sweep + self.sweep_rate * steptime, self.sweep_limits[0], self.sweep_limits[1])
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        self.throttle = 0.0
