import numpy as np

# Perching scenario

state_dims = 8
actions = 49
failReward = 0.0
h_min = -15

def wrap_class(BixlerClass, parameters):
    class PerchingBixler(BixlerClass):
                def __init__(self):

                    super(PerchingBixler,self).__init__(parameters)
                    self.var_start = parameters.get("variable_start")

                def is_out_of_bounds(self):
                    def is_in_range(x,lower,upper):
                        return lower < x and x < upper
                    # Prevent flipping over
                    if self.orientation_e[1,0] > np.pi / 2:
                        return True
                    # Check x remains sensible
                    if not is_in_range(self.position_e[0,0],-50,10):
                        return True
                    # Check y remains sensible
                    if not is_in_range(self.position_e[1,0],-2,2):
                        return True
                    # Check z remains sensible (i.e. not crashed)
                    if not is_in_range(self.position_e[2,0],h_min,1):
                        return True
                    # Check u remains sensible, > 0
                    if not is_in_range(self.velocity_b[0,0],0,20):
                        return True
                    # airspeed remains sensible
                    if self.airspeed > 100:
                        return True    
                    return False
        
                # def get_reward(self):
                #     if self.is_terminal():
                #         if self.is_out_of_bounds():
                #             return failReward
                #         cost_vector = np.array([10,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0])
                #         cost = np.dot( np.squeeze(self.get_state()) ** 2, cost_vector ) / 2500
                #         # product_list = [a*b for a,b in zip((np.squeeze(self.get_state()) ** 2), cost_vector)]
                #         # product_list = [a/2500 for a in product_list]
                #         # mask = [0,2,4,6,8]
                #         # product_list = [product_list[i] for i in mask]
                #         # product_list = [1/(sum(product_list)/a) for a in product_list]
                #         # print(product_list)
                #         return  ((1.0 - cost) * 2.0) - 1.0
                #     return 0.0

                 #     return 0.0
                @staticmethod
                def gaussian(x, sig = 0.4, mu = 0):   
                           return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
                
                def get_reward(self):
                    if self.is_terminal():
                        if self.is_out_of_bounds():
                            return failReward
                        obs = np.array([self.position_e[0,0], self.position_e[2,0], self.orientation_e[1,0], self.velocity_b[0,0], self.velocity_b[2,0]])
                        target_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype='float64')
                        bound = np.array([15, 5, np.deg2rad(20),10,10])
                        cost = (target_state - obs)/bound
                        cost = list(map(self.gaussian, cost))
                        reward = np.prod(cost)
                        return reward
                    return 0.0
        
                def is_terminal(self):
                    # Terminal point is floor
                    if self.position_e[2,0] > 0:
                        return True
                    return self.is_out_of_bounds()

                def get_state(self):
                    return super(PerchingBixler,self).get_state()[0:14].T


                def get_normalized_obs(self, state=None):
                    if state is None:
                        state=self.get_state()

                    obs = np.float64(np.delete(state, [1, 3, 5, 7, 9, 11], axis=1))

                    pb2 = np.pi*2

                    mins = np.array([ -50,  h_min,  -pb2, -10, -10,  -pb2,  self.sweep_limits[0], self.elev_limits[0]])
                    maxs = np.array([  10,       1,  pb2, 20,   10,   pb2,  self.sweep_limits[1], self.elev_limits[1]])

                    return (obs-mins)/(maxs-mins)
                
                def reset_scenario(self):
                    
                    self.wind_sim.update()
                    wind = self.wind_sim.get_wind()

                    target_airspeed = 13 # m/s
                    u = target_airspeed + wind[0]

                    initial_state = np.array([[-40,0,-5, 0,0,0, u,0,0, 0,0,0, 0,0,0]], dtype="float64")

                    if self.var_start:
                        # Add noise to starting velocity
                        # start_shift_u =  np.random.uniform(-1.0, 1.0)
                        # start_shift_w = np.random.uniform(-1.0, 1.0)
                        # start_shift_theta = np.random.uniform(-0.061, 0.061) #shift +-3.5degs in theta

                        start_shift_u =  self.np_random.uniform(-1.0, 1.0)
                        start_shift_w = self.np_random.uniform(-1.0, 1.0)
                        start_shift_theta = self.np_random.uniform(-0.061, 0.061) #shift +-3.5degs in theta

                        # Scale for +- 1m/s
                        initial_state[:,6] += start_shift_u
                        initial_state[:,8] += start_shift_w
                        initial_state[:,4] += start_shift_theta

                    self.set_state(initial_state)

                    self.wind_sim.update()


    return PerchingBixler
