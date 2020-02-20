import numpy as np
# from wind.dryden import Dryden

class Wind(object):

    def __init__(self, steady_var = False, steady_state = [0, 0, 0], dryden_on = False, min_mag = 0, max_mag = 0):

        self.dryden_on = dryden_on
        self.min = min_mag
        self.max = max_mag

        if steady_var:
            # TODO
            pass
        else:
            # TODO make array np array from list
            self.steady_state = np.array((steady_state))
        
        if dryden_on == True:

            pass

    
    def update(self):

        return self.steady_state

        # dryden.update()

        # return np.concatenate(self.steady_state, dryden.update(ts))



    def seed(self, np_random):

        self.np_random = np_random 

    def update_wind(self):

        return self.steady_state

    

    def get_steady_state(self):

        return self.steady_state

    















