import numpy as np
import controllers.common as common

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_SweepElevatorCont(common.BixlerController):
    
    def __init__(self,noise=0.0):
        super(Bixler_SweepElevatorCont, self).__init__(noise)
        
    def set_action(self,action):
        
        self.sweep = action[0]
        self.elev = action[1]
    
    def update_control_surfaces(self,steptime):
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        self.throttle = 0.0
