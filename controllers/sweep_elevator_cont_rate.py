import numpy as np
import controllers.common as common

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_SweepElevatorContRate(common.BixlerController):
    
    def __init__(self, parameters):
        super(Bixler_SweepElevatorContRate, self).__init__(parameters)

        # Control surface rates for each action
        self.elev_rate_lim = 60
        self.sweep_rate_lim = 60
        
        # Control surface rates
        self.sweep_rate = 0 # (deg/s)
        self.elev_rate = 0  # (deg/s)
        
    def set_action(self,action):
        
        self.elev_rate =  action[0] * self.elev_rate_lim
        self.sweep_rate = action[1] * self.sweep_rate_lim

    def update_control_surfaces(self,steptime):
        self.sweep = np.clip(self.sweep + self.sweep_rate * steptime, self.sweep_limits[0], self.sweep_limits[1])

        # print(self.sweep)
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])

        # print(self.elev)
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        self.throttle = 0.0
