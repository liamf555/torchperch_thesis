import numpy as np
import controllers.common as common
import time

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_SweepElevator(common.BixlerController):
    
    def __init__(self, parameters):
        super(Bixler_SweepElevator, self).__init__(parameters)
        
        # Control surface rates for each action
        self.elev_rates = [-60, -10, -5, 0, 5, 10, 60]
        self.sweep_rates = [ -60, -10, -5, 0, 5, 10, 60 ]
        
        # Control surface rates
        self.sweep_rate = 0 # (deg/s)
        self.elev_rate = 0  # (deg/s)
        self.latency_params = parameters.get("latency") # (sec)
        self.latency = 0
        self.next_sweep_rate = 0
        self.next_elev_rate = 0
        self.time_since_action = 0
        self.prev_sweep = 0

    def set_action(self,action_index):
        # Action may be control surface rates / throttle
        elev_rate_idx = int(action_index) // 7
        sweep_rate_idx = int(action_index) % 7

        # self.elev_rate =  self.elev_rates[elev_rate_idx]
        # self.sweep_rate = self.sweep_rates[sweep_rate_idx]

        self.next_elev_rate = self.elev_rates[elev_rate_idx]
        self.next_sweep_rate = self.sweep_rates[sweep_rate_idx]
        self.time_since_action = 0
        # self.latency = 0.02 +  np.random.normal(0, 0.0025)
        self.latency = 0.0
    
    def update_control_surfaces(self,steptime):

        self.time_since_action += steptime

        # print(self.time_since_action)
        # print(self.latency)

        if (self.time_since_action >= self.latency):
            self.sweep_rate = self.next_sweep_rate
            self.elev_rate = self.next_elev_rate

        self.sweep = np.clip(self.sweep + self.sweep_rate * steptime, self.sweep_limits[0], self.sweep_limits[1])
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])

        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        self.throttle = 0.0
        
