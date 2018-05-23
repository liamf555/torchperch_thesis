import bixler
import math

class BixlerController(bixler.Bixler):
    def __init__(self,noise=0.0):
        super().__init__(noise)
    
    def step(self,steptime):
        # Bixler model stability needs timesteps of < 0.01s
        base_steptime = 0.01
        step_fraction, whole_steps = math.modf( steptime / base_steptime )
        try:
            for i in range(int(whole_steps)):
                self.update_control_surfaces(base_steptime)
                super().step(base_steptime)
            self.update_control_surfaces(base_steptime*step_fraction)
            super().step(base_steptime*step_fraction)
        except Exception as e:
            raise
