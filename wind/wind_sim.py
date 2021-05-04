import numpy as np
# import wandb

class Wind(object):

    def __init__(self, wind_mode = None, wind_params = None, turbulence = False):

        self.wind_mode = wind_mode

        self.wind_params = wind_params
        self.wind_vector = [0, 0, 0]
        self.turbulence = turbulence

        # if self.turbulence:
        #     self.dryden = DrydenGustModel(Va = 13, intensity=self.turbulence)
        #     self.dryden.simulate(20)
        #     self.update()
    
    def update(self):

        if self.wind_mode == 'normal':
            wind_north = self.np_random.normal(self.wind_params[0], self.wind_params[1])
            self.wind_vector = [wind_north, 0, 0]

            # wandb.log({"wind_speed": self.wind_vector[0]})

        if self.wind_mode == 'normal_eval' or self.wind_mode == 'steady_eval':
            self.wind_vector = self.wind_params
            

        if self.wind_mode == 'steady':
            self.wind_vector = self.wind_params
            

        if self.wind_mode == 'uniform':
            wind_north = np.random.uniform(self.wind_params[0], self.wind_params[1])
            self.wind_vector = [wind_north, 0, 0]

            # print(self.wind_vector)
            
            # wandb.log({"wind_speed": self.wind_vector[0]})

        # if self.turbulence:
        #     self.dryden.reset()
        #     self.dryden.simulate(20)
        #     self.gusts = self.dryden.vel_lin.tolist()
        #     # print(self.dryden.vel_lin)
        #     # print(self.gusts)

    # def get_wind(self):
    
    def get_wind(self):

        # if self.turbulence:
        #     dryden = np.array([[self.gusts[0].pop(0), 0.0, self.gusts[2].pop(0)]]) 
        #     wandb.log({"dryden_u": dryden[0]}) 
        #     return (np.array([self.wind_vector]), dryden)

        return np.array([self.wind_vector])

    

    def seed(self, seed=None):

        self.np_random = np.random.RandomState(seed)
       
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

        
        






# def get_wind(self):

#         if self.wind_mode == 'evaluate_normal':
#             print(self.wind_vector)

#         return self.wind_vector

    















