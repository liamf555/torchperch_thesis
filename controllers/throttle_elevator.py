import numpy as np
import controllers.common as common

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_ThrottleElevator(common.BixlerController):
    
    def __init__(self,parameters):
        super(Bixler_ThrottleElevator, self).__init__(parameters)
        
        # Control surface rate / throttle settings per action
        self.elev_rates = [ -60, -10, -5, 0, 5, 10, 60 ]
        self.throttles = [ 0, 2, 4, 6, 8, 10, 12]
        
        # Control surface rates (internal state)
        self.elev_rate = 0 # (deg/s)

    def set_action(self,action_index):
        elev_rate_idx = int(action_index) % 7
        throttle_idx = int(action_index) // 7

        # print(elev_rate_idx)

        self.elev_rate = self.elev_rates[elev_rate_idx]
        self.throttle = self.throttles[throttle_idx]
        # print(self.throttle)

    def update_control_surfaces(self,steptime):
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])
        self.sweep = 0.0
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
