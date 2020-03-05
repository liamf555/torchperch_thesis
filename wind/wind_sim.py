import numpy as np
# from wind.dryden import Dryden

class Wind(object):

    def __init__(self, steady_var = False, steady_state = [0, 0, 0], dryden_on = False):

        self.dryden_on = dryden_on

        self.steady_var = steady_var 

        if steady_var:
            self.max = steady_state[0]
            self.min = - steady_state[0]

            self.steady_state = np.array((steady_state))
        
        if dryden_on == True:

            pass

    
    def update(self):

        if self.steady_var:

            wind_north = np.random.normal(2.58, 1.526)
            self.steady_state[0] = wind_north

    def seed(self, np_random):

        self.np_random = np_random 

    def get_wind(self):

        return self.steady_state

    















