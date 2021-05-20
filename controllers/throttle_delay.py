import numpy as np
import controllers.common as common
import wandb

# Bixler wrapper to set what control surfaces are utilised by the NN
class Bixler_ThrottleDelay(common.BixlerController):
    
    def __init__(self, parameters):
        super(Bixler_ThrottleDelay, self).__init__(parameters)
        
        self.elev_rates = [-60, -10, -5, 0, 5, 10, 60]
        self.sweep_rates = [ -60, -10, -5, 0, 5, 10, 60 ]
        
        # Control surface rates
        self.sweep_rate = 0 # (deg/s)
        self.elev_rate = 0  # (deg/s)
        self.next_sweep_rate = 0
        self.next_elev_rate = 0
        self.throttle_on = True
        self.throttle_change = 0
        self.prev_throttle = True
        self.change_time = 0
        self.time_log = True

        #Latency params
        self.latency_on = parameters.get("latency") # (sec)

    def set_action(self,action):
        # print(action)
        # Action may be control surface rates / throttle
    
        self.next_elev_rate = self.elev_rates[action[0]]
        self.next_sweep_rate = self.sweep_rates[action[1]]

        self.time_since_action = 0.0
        if self.latency_on:
            self.latency = (18 + np.random.lognormal(0, 1))/1000
        else:
            self.latency = 0.0

        # print(action[2])

        if action[2] == 0:
            self.throttle_on = False
        elif action[2] == 1:
            self.throttle_on = True

        self.change_time += 0.1

        if (self.prev_throttle == True and self.throttle_on == False) or (self.prev_throttle == False and self.throttle_on == True):
            self.throttle_change += 1
            if self.throttle_change == 1 and self.time_log == True:
                wandb.log({"change_time": self.change_time})
                self.time_log = False 
        
        self.prev_throttle = self.throttle_on

    def update_control_surfaces(self,steptime):

        self.time_since_action += steptime

        if (self.time_since_action >= self.latency):
            self.sweep_rate = self.next_sweep_rate
            self.elev_rate = self.next_elev_rate

        self.sweep = np.clip(self.sweep + self.sweep_rate * steptime, self.sweep_limits[0], self.sweep_limits[1])
        self.elev = np.clip(self.elev + self.elev_rate * steptime, self.elev_limits[0], self.elev_limits[1])

        if self.throttle_on:
            self.throttle = 2.37
        else:
            self.throttle = 0
                 
        self.rudder = 0.0
        self.tip_port = 0.0
        self.tip_stbd = 0.0
        self.washout = 0.0
        
