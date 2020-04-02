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

            wandb.log({"wind_speed": self.wind_vector[0]})

        if self.wind_mode == 'normal_eval' or self.wind_mode == 'steady_eval':
            self.wind_vector = self.wind_params
            

        if self.wind_mode == 'steady':
            self.wind_vector = self.wind_params

            wandb.log({"wind_speed": self.wind_vector[0]}) 

        


    def get_wind(self):

        return self.wind_vector

def make_eval_wind(wind_mode, wind_params):

    wind_north = []
    if wind_mode == 'normal_eval':
        mean, sd = wind_params
        wind_north = [mean - (2 * sd), mean, mean + (2 * sd)]
    
    if wind_mode == "steady_eval":
        # wind_params = ([wind_params[i:i + 3] for i in range(0, len(wind_params), 3)])
        wind_north = wind_params

    wind_north = [float(i) for i in wind_north]

    return wind_north

        
        


        










    def seed(self, np_random):

        self.np_random = np_random 

    def get_wind(self):

        if self.wind_mode == 'evaluate_normal':
            print(self.wind_vector)

        return self.wind_vector

    















