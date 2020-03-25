import numpy as np
import wandb
# from wind.dryden import Dryden

class Wind(object):

    def __init__(self, wind_mode = None, wind_params = None, dryden_on = False):

        self.wind_mode = wind_mode

        self.wind_params = wind_params
        self.wind_vector = [0, 0, 0]




        
        
    
    def update(self):

        if self.wind_mode == 'normal':

            wind_north = np.random.normal(self.wind_params[0], self.wind_params[1])
            self.wind_vector = [wind_north, 0, 0]

        if self.wind_mode == 'evaluate_normal':
            self.wind_vector = self.wind_params
            # print(self.wind_vector)
        
        wandb.log({"wind_speed": self.wind_vector[0]})
        


    def seed(self, np_random):

        self.np_random = np_random 

    def get_wind(self):

        # if self.wind_mode == 'evaluate_normal':
        #     print(self.wind_vector)

        return self.wind_vector

    















