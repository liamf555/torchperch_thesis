import numpy as np

# Perching scenario

state_dims = 7
actions = 7
failReward = 0.0
h_min = -30
stall_speed = 9

def wrap_class(BixlerClass, parameters):
    class PerchingBixler(BixlerClass):
                def __init__(self):

                    super(PerchingBixler,self).__init__(parameters)
                    self.var_start = parameters.get("variable_start")
                
                @staticmethod
                def _is_in_range(x,lower,upper):
                        return lower < x and x < upper

                def is_out_of_bounds(self):
                    # Prevent flipping over
                    if self.orientation_e[1,0] > np.pi:
                        return True
                    # Check x remains sensible
                    if not self._is_in_range(self.position_e[0,0],-10, 30 ):
                        return True
                    # Check y remains sensible
                    if not self._is_in_range(self.position_e[1,0],-2,2):
                        return True
                    # Check z remains sensible (i.e. not crashed)
                    if not self._is_in_range(self.position_e[2,0],h_min,1):
                        return True
                    # Check u remains sensible, > 0
                    if not self._is_in_range(self.velocity_b[0,0],(0.5 * stall_speed), 3 * stall_speed):
                        return True
                    return False
        
                # def get_reward(self):
                #     if self.is_terminal():
                #         if self.is_out_of_bounds():
                #             return failReward
                #         target_state = np.array([0,0,0, 0, 0 ,0, (1.1 * stall_speed), 0, 0, 0,0,0, 0, 0], dtype='float64')
                #         cost_vector = np.array([0,0,0, 0,100,0, 10,0,1, 0,0,0, 0,0])
                #         scaling = np.array([0, 0,0, 0, np.pi / 2, 0, 20, 0, 10, 0,0,0, 0,0 ])
                #         norm = np.dot(scaling**2, cost_vector)
                #         cost = np.dot( np.squeeze((self.get_state()) - target_state)** 2, cost_vector) / norm
                #         height_reward = (-10 - self.position_e[2,0]) / 10
                #         return  ((1 - cost) * 2) - 1 + height_reward
                #     return 0.0

                def get_reward(self):
                    if self.is_terminal():
                        if self.is_out_of_bounds():
                            return failReward
                        def gaussian(x, sig = 0.4, mu = 0):   
                            return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
                        obs = np.array([self.orientation_e[1,0], self.velocity_b[0,0], self.velocity_b[2,0]])
                        target_state = np.array([0, 1.1*stall_speed, 0], dtype='float64')
                        bound = np.array([np.deg2rad(20),5,5])
                        cost = (target_state - obs)/bound
                        # print(cost)
                        cost = list(map(gaussian, cost))
                        # print(cost)
                        cost = np.prod(cost)
                        # print(cost)
                        height_reward = (-10 - self.position_e[2,0])
                        reward = np.clip((cost * height_reward), 0, None)
                        # print(reward)
                        return reward
                    return 0.0
                    


        
                def is_terminal(self):
                    # Terminal point is floor
                    if self.position_e[0, 0] > 50:
                        return True
                    if self.velocity_b[0, 0] < (1.1 * stall_speed):
                        return True
                    return self.is_out_of_bounds()

                def get_state(self):
                    return super(PerchingBixler,self).get_state()[0:14].T


                def get_normalized_obs(self, state=None):
                    if state is None:
                        state=self.get_state()

                    obs = np.float64(np.delete(state, [1, 3, 5, 7, 9, 11, 12], axis=1))

                    # pb2 = np.pi*2

                    # mins = np.array([ -20,  h_min,  -pb2, -10, -10,  -pb2, self.elev_limits[0]])
                    # maxs = np.array([  20,       1,  pb2, 20,   10,   pb2, self.elev_limits[1]])


                    return obs

                    # return (obs-mins)/(maxs-mins)
                
                def reset_scenario(self):
                    
                    self.wind_sim.update()
                    wind = self.wind_sim.get_wind()

                    # target_airspeed = 13 # m/s
                    # u = target_airspeed + wind[0]

                    initial_speed = stall_speed * 2.0

                    initial_state = np.array([[0,0,-10, 0,0,0, initial_speed,0,0, 0,0,0, 0,0,0]], dtype="float64")

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


    return PerchingBixler
