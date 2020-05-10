import numpy as np
import controllers.common as common

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_ElevatorCont(common.BixlerController):
    
    def __init__(self,parameters):
        super(Bixler_ElevatorCont, self).__init__(parameters)
        
        # Control surface rate  per action
        self.elev_rate_lim = 120
        
        # Control surface rates (internal state)
        self.elev_rate = 0 # (deg/s)

    def set_action(self,action):
        
        self.elev_rate =  action[0] * self.elev_rate_lim
   
    def update_control_surfaces(self,steptime):
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])
        self.sweep = 0.0
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        self.throttle = 0.0

