import numpy as np
try:
    import wandb
except ImportError:
    pass

# Perching scenario

state_dims = 14
actions = 49
failReward = 0.0
h_min = -500
stall_speed = 9

def wrap_class(BixlerClass, parameters):
    class PerchingBixler(BixlerClass):
                def __init__(self):
                    super(PerchingBixler,self).__init__(parameters)
                    self.var_start = parameters.get("variable_start")
                    self.target = np.array([ (np.sqrt(2)/2) * 30 , -(130 - (np.sqrt(2)/2) * 30)])
                    # self.target = np.array([0 , -100])
                    # self.target = np.array([30, -130])
                    # self.target = np.array([ (np.sqrt(2)/2) * 25 , -(125 - (np.sqrt(2)/2) * 25) , (np.pi/4)])
                    # self.target = np.array([0, -150, (np.pi)])
                    # self.target = np.array([-25, -125, np.deg2rad(270)])
                    # self.target = np.array([np.deg2rad(40)])

                    self.sector_switch = 1
                    self.done_flag = False
                    self.reward_flag = False
                    self.time = 0
                    self.reward = 0
                    self.radius = 30
                
                @staticmethod
                def _is_in_range(x,lower,upper):
                        return lower < x and x < upper

                @staticmethod
                def gaussian(x, sig = 0.4, mu = 0): 
                    try:  
                        return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
                    except FloatingPointError:
                        return 0.0
                   
                def is_out_of_bounds(self):
                    # Prevent flipping over
                    # if self.orientation_e[1,0] < 0:
                    #     return True
                    if self.omega_b[1,0] < -5:
                        return True
                    # if self.omega_b[1,0] > 100:
                        # return True
                    # if self.time > 10:
                    #     return True
                    # Check x remains sensible
                    if not self._is_in_range(self.position_e[0,0],-50, 100):
                        return True
                    # Check y remains sensible
                    if not self._is_in_range(self.position_e[1,0],-2,2):
                        return True
                    # Check z remains sensible (i.e. not crashed)
                    if not self._is_in_range(self.position_e[2,0],h_min,1):
                        return True
                    # Check u remains sensible, > 0
                    if not self._is_in_range(self.velocity_b[0,0],(stall_speed), 3 * stall_speed):
                        return True
                    return False
        
                # def get_reward(self):
                #     if self.is_terminal():
                #         if self.is_out_of_bounds():
                #             return failReward
                #     final_theta = self.orientation_e[1,0]
                #     # cost = ((self.target_theta - final_theta)**2) / (np.pi **2)
                #     if final_theta < np.pi:
                #         self.target_theta = 0
                #         obs = np.array([self.position_e[2,0], final_theta])
                #         target_state = np.array([100, self.target_theta], dtype='float64')
                #         bound = np.array([15, np.deg2rad(20)])
                #         cost = (target_state - obs)/bound
                #         reward = np.prod(cost)
                #         return reward
                #     else:
                #     # print(f'final_theta: {np.rad2deg(final_theta)}')
                #         cost = ((self.target_theta - final_theta)**2) / (np.pi **2)
                #     # print(cost)
                #         return  ((1 - cost) * 2) - 1
                #     # return 0.0

                def get_reward(self):
                    self.time += 0.1
                    if self.is_terminal():
                        if self.is_out_of_bounds():
                            return failReward
                    #     cost_vector = np.array([1, 1, 1])
                    #     scaling = np.array([80, 80, 20])
                    #     norm = np.dot(scaling**2, cost_vector)
                    #     current_state = np.array([self.position_e[0,0], self.position_e[2,0], self.velocity_b[0,0]])
                    #     cost = np.dot((self.target - current_state) ** 2, cost_vector) / norm
                    #     reward =   ((1.0 - cost) * 2.0) - 1.0
                    #     return reward
                    # return 0.0
                    # if self.reward_flag: 
                    #     if self.sector_switch == 3:
                    #         current_state = np.array([self.position_e[0,0], self.position_e[2,0], self.velocity_b[0,0]])
                    #         # print(current_state)
                    #         bound = np.array([20, 40, 20])
                    
                    current_state = np.array([self.position_e[0,0], self.position_e[2,0]])
                    # print(current_state)
                    bound = np.array([20, 40])
                    cost = (self.target - current_state)/bound
                    # print(cost)
                    cost = list(map(self.gaussian, cost))
                    # print(cost)
                    cost = np.prod(cost)

                    return cost
                    # self.reward += cost
                    #     self.reward_flag = False
                    #     if self.done_flag == True:
                    #         return self.reward
                    # return 0.0

                    #         # print(current_state)
                    #         self.reward += reward
                    #         # print(reward, self.reward)
                    #         self.reward_flag = False
                    #     if self.done_flag == True:
                    #         return self.reward
                    # return 0.0

                # def get_reward(self):
                #     self.time += 0.1
                #     if self.is_terminal():
                #         if self.is_out_of_bounds():
                #             return failReward
                #     # theta = self.orientation_e[1,0] if self.orientation_e[1,0] > 0 else ((2*np.pi) + self.orientation_e[1,0])
                #     current_state = np.array(self.omega_b[1,0])
                #     if abs(current_state) > np.deg2rad(1000):
                #         return failReward
                    
                #     if current_state < 0:
                #         return failReward
                #     # cost = (self.target - current_state) ** 2
                #     cost_vector = np.array([1])
                #     scaling = np.array([np.deg2rad(80)])
                #     norm = np.dot(scaling**2, cost_vector)
                #     # current_state = np.array([self.position_e[0,0], self.position_e[2,0]])
                #     cost = np.dot((self.target - current_state) ** 2, cost_vector) / norm
                #     reward =  ((1.0 - cost) * 2.0) - 1.0
                #     reward = np.clip(reward, 0, 1.0)
                #     # print(reward)
                #     return reward

                    # print(np.rad2deg(current_state))
                    # print(self.throttle)
                    # print(cost[0])
                    # return -cost[0]
                    
                    # return  ((1.0 - cost) * 2.0) - 1.0
                  

                

                    

                # def get_reward(self):
                #     if self.is_terminal():
                #         if self.is_out_of_bounds():
                #             return failReward
                #     cost_vector = np.array([1, 1])
                #     scaling = np.array([80, 80])
                #     norm = np.dot(scaling**2, cost_vector)
                #     current_state = np.array([self.position_e[0,0], self.position_e[2,0]])
                #     cost = np.dot((self.target - current_state) ** 2, cost_vector) / norm
                #     return  ((1.0 - cost) * 2.0) - 1.0
                

                
                def _get_target(self):
                    theta = self.orientation_e[1,0] if self.orientation_e[1,0] > 0 else ((2*np.pi) + self.orientation_e[1,0])
                    if self.sector_switch == 1:
                        if (np.deg2rad(45)) < theta < np.deg2rad(135):
                            self.target = np.array([30, -130])
                            # self.target = np.array([25, -125, np.pi])
                            self.sector_switch = 2
                            self.reward_flag = True
                    if self.sector_switch == 2:
                        if (np.pi/2) < theta < np.pi:
                            self.target = np.array([(np.sqrt(2)/2) * 30 , -(130 + (np.sqrt(2)/2) * 30)])
                            # self.target = np.array([(np.sqrt(2)/2) * 25 , -(125 + (np.sqrt(2)/2) * 25) , (np.pi/4)])
                            self.sector_switch = 3
                            self.reward_flag = True
                    if self.sector_switch == 3:
                        if np.deg2rad(135) < theta < np.deg2rad(225):
                            self.target = np.array([0, -160])
                            # self.target = np.array([0, -150, np.pi])
                            self.sector_switch = 4
                            self.reward_flag = True
                    if self.sector_switch == 4:
                        if np.pi < theta < np.deg2rad(270):
                            self.target = np.array([-(np.sqrt(2)/2) * 30, -(130 + (np.sqrt(2)/2) * 30)])
                            self.sector_switch = 5
                            self.reward_flag = True
                    if self.sector_switch == 5:
                        if np.deg2rad(225) < theta < np.deg2rad(315):
                            self.target = np.array([-25, -125])
                            self.sector_switch = 6
                            self.reward_flag = True
                    if self.sector_switch == 6:
                        if np.deg2rad(270) < theta < np.deg2rad(360):
                            self.target = np.array([-(np.sqrt(2)/2) * 30, -(130 - (np.sqrt(2)/2) * 30)])
                            self.sector_switch = 7
                            self.reward_flag = True
                    if self.sector_switch == 7:
                        if np.deg2rad(315) < theta < np.deg2rad(45):
                            self.target = np.array([0, -100])
                            self.sector_switch = 8
                            self.reward_flag = True
                    if self.sector_switch == 8:
                        if np.deg2rad(0) < theta < np.deg2rad(90):
                            self.reward_flag = True
                            self.done_flag = True

                        

                    #         self.target = np.array([0, -100, 0])
                    #         self.sector_switch = 4
                    # if self.sector_switch == 4:
                    #     if 0 < theta < (np.pi/2):
                    #         self.done_flag = True

                # def _get_target(self):
                #     theta = self.orientation_e[1,0] if self.orientation_e[1,0] > 0 else ((2*np.pi) + self.orientation_e[1,0])
                #     if self.sector_switch == 1:
                #         if (np.deg2rad(45)) < theta < np.deg2rad(65):
                # #             self.target = np.array([25, -125])
                # #             # self.target = np.array([25, -125, np.pi])
                #             self.sector_switch = 2
                #     if self.sector_switch == 2:
                #         if np.deg2rad(90) < theta < np.deg2rad(135):
                #             self.target = np.array([0, -160, stall_speed*1.1])
                #             self.sector_switch = 3
                #             self.reward_flag = True
                #     if self.sector_switch == 3:
                #         if np.deg2rad(180) < theta < np.deg2rad(225):
                #             self.reward_flag = True
                #             self.done_flag = True

                            # self.done_flag = True

                    




                            



                # def get_reward(self):
                #     if self.is_terminal():
                #         if self.is_out_of_bounds():
                #             return failReward
                #         def gaussian(x, sig = 0.4, mu = 0):   
                #             return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
                #         obs = np.array([self.orientation_e[1,0], self.velocity_b[0,0], self.velocity_b[2,0]])
                #         target_state = np.array([0, 1.1*stall_speed, 0], dtype='float64')
                #         bound = np.array([np.deg2rad(20),5,5])
                #         cost = (target_state - obs)/bound
                #         # print(cost)
                #         cost = list(map(gaussian, cost))
                #         # print(cost)
                #         cost = np.prod(cost)
                #         # print(cost)
                #         height_reward = (-10 - self.position_e[2,0])
                #         reward = np.clip((cost * height_reward), 0, None)
                #         # print(reward)
                #         return reward
                #     return 0.0
                    

                def is_terminal(self):
                    # Terminal point is floor
                    # if self.position_e[2, 0] > 0:
                    #     return True
                    # # if self.position_e[2,0] < -140:
                    #     return True
                    # if self.velocity_b[0,0] < stall_speed:
                        # return True
                    if self.position_e[0, 0] > 100:
                        self.reward_flag = True
                        self.done_flag = True
                        return True
                    if self.position_e[0, 0] < - 100:
                        self.reward_flag = True
                        self.done_flag = True
                        return True
                    # if self.prev_theta > np.pi/2 :
                    self._get_target()
                    # print(self.done_flag)
                    if self.done_flag == True:
                        self.reward_flag = True
                        return True
                    if self.time > 10:
                        self.reward_flag = True
                        self.done_flag = True
                        return True
                    # if self.prev_theta < np.pi/2 :
                    # if self.orientation_e[1,0] < 0:
                    # return True
                    # self.prev_theta = self.orientation_e[1,0]
                    # print(self.done_flag)


                
                def get_state(self):
                    return super(PerchingBixler,self).get_state()[0:14].T


                def get_normalized_obs(self, state=None):
                    if state is None:
                        state=self.get_state()

                    # obs = np.float64(np.delete(state, [1, 3, 5, 7, 9, 11, 12], axis=1))

                    # pb2 = np.pi*2

                    # mins = np.array([ -20,  h_min,  -pb2, -10, -10,  -pb2, self.elev_limits[0]])
                    # maxs = np.array([  20,       1,  pb2, 20,   10,   pb2, self.elev_limits[1]])

                    return state

                    # return (obs-mins)/(maxs-mins)
                
                def reset_scenario(self):
                    
                    self.wind_sim.update()
                    wind = self.wind_sim.get_wind()

                    # target_airspeed = 13 # m/s
                    # u = target_airspeed + wind[0]

                    initial_speed = 13

                    initial_state = np.array([[0, 0, -100, 0, 0, 0, initial_speed, 0, 0, 0,0, 0, 0, 0,0]], dtype="float64")
                    self.throttle = 0

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

                    self.done_flag = False
                    self.time = 0
                    self.reward = 0
                    self.sector_switch = 1
                    self.target = np.array([ (np.sqrt(2)/2) * 30 , -(130 - (np.sqrt(2)/2) * 30)])

    return PerchingBixler