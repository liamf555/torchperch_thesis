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
        self.next_sweep_rate = 0
        self.next_elev_rate = 0

        #Latency params
        self.latency_on = parameters.get("latency")
        
    def set_action(self,action):

        self.next_elev_rate = action[0] * self.elev_rate_lim
        self.next_sweep_rate = action[1] * self.sweep_rate_lim
    
        self.time_since_action = 0.0 
        if self.latency_on:
            self.latency = (18 + np.random.lognormal(0, 1))/1000
        else:
            self.latency = 0.0

    def update_control_surfaces(self,steptime):

        self.time_since_action += steptime
 
        if (self.time_since_action >= self.latency):
            self.sweep_rate = round(self.next_sweep_rate)
            self.elev_rate = round(self.next_elev_rate)

        self.sweep = np.clip(self.sweep + self.sweep_rate * steptime, self.sweep_limits[0], self.sweep_limits[1])
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])

        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        self.throttle = 0.0
