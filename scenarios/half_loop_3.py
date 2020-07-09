import numpy as np
import wandb

# Perching scenario

state_dims = 14
actions = 49
failReward = -1.0
h_min = -500
stall_speed = 9

def wrap_class(BixlerClass, parameters):
    class PerchingBixler(BixlerClass):
                def __init__(self):
                    super(PerchingBixler,self).__init__(parameters)
                    self.var_start = parameters.get("variable_start")
                    # self.target = np.array([ (np.sqrt(2)/2) * 25 , -(125 - (np.sqrt(2)/2) * 25)])
                    # self.target = np.array([0 , -100])
                    # self.target = np.array([ (np.sqrt(2)/2) * 25 , -(125 - (np.sqrt(2)/2) * 25) , (np.pi/4)])
                    # self.target = np.array([0, -150, (np.pi)])
                    # self.target = np.array([-25, -125, np.deg2rad(270)])
                    # self.target = np.array([np.deg2rad(40)])
                    self.target = np.array([0, -155, np.pi, stall_speed*1.1])
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
                    # if abs(self.alpha) > 100
                    # Check x remains sensible
                    if not self._is_in_range(self.position_e[0,0],-50, 200):
                        print(self.position_e[0,0], 'x')
                        return True
                    # Check y remains sensible
                    if not self._is_in_range(self.position_e[1,0],-2,2):
                        return True
                    # Check z remains sensible (i.e. not crashed)
                    if not self._is_in_range(self.position_e[2,0],h_min,1):
                        return True
                    # Check u remains sensible, > 0
                    if not self._is_in_range(self.velocity_b[0,0],stall_speed, 3 * stall_speed):
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
                    # print(self.time)
                    if self.is_terminal():
                        if self.is_out_of_bounds():
                            return failReward
                    cost_vector = np.array([1, 1, 1, 1])
                    scaling = np.array([80, 80, np.deg2rad(90), 20])
                    norm = np.dot(scaling**2, cost_vector)
                    current_state = np.array([self.position_e[0,0], self.position_e[2,0],  self.orientation_e[1,0], self.velocity_b[0,0]])
                    cost = np.dot((self.target - current_state) ** 2, cost_vector) / norm
                    reward =   ((1.0 - cost) * 2.0) - 1.0
                    return reward
                    # return 0.0
                    # print(self.get_state())
                    # if self.reward_flag: 
                    # current_state = np.array([self.position_e[0,0], self.position_e[2,0], self.orientation_e[1,0], self.velocity_b[0,0]])
                    # bound = np.array([20, 40, np.deg2rad(90), 20])
                    # cost = (self.target - current_state)/bound
                    # # print(current_state, self.target)
                    # # print(cost)
                    # cost = list(map(self.gaussian, cost))
                    # # print(cost)
                    # cost = np.prod(cost)

                    # return cost
                    # self.reward += cost
                    # self.reward_flag = False
                    # if self.done_flag:
                    #     return self.reward
                    # return 0.0

        
                def _get_target(self):
                    theta = self.orientation_e[1,0] if self.orientation_e[1,0] > 0 else ((2*np.pi) + self.orientation_e[1,0])
                    if self.sector_switch == 1:
                        if self.position_e[0,0] > 15:
                        # if (np.deg2rad(45)) < theta < np.deg2rad(65):
                        # if np.deg2rad(180) < theta < np.deg2rad(225):
                            # self.reward_flag = True
                            # self.done_flag = True
                            print('target')
                #             self.target = np.array([25, -125])
                #             # self.target = np.array([25, -125, np.pi])
                            self.sector_switch = 2
                    if self.sector_switch == 2:
                         if self.position_e[0,0]< 0:
                             self.done_flag = True
                    #     if np.deg2rad(90) < theta < np.deg2rad(135):
                    #         self.target = np.array([0, -160, stall_speed*1.1])
                    #         self.sector_switch = 3
                    #         self.reward_flag = True
                    # if self.sector_switch == 2:
                    #     if np.deg2rad(180) < theta < np.deg2rad(225):
                    #         self.reward_flag = True
                    #         self.done_flag = True
                    #         print('target')

                def is_terminal(self):
                    if self.position_e[0, 0] > 50:
                        # self.reward_flag = True
                        # self.done_flag = True
                        print('pos')
                        return True
                    if self.position_e[0, 0] < 0:
                        # self.reward_flag = True
                        # self.done_flag = True
                        return True
                    # if self.prev_theta > np.pi/2 :
                    self._get_target()
                    # print(self.done_flag)
                    if self.done_flag == True:
                        # self.reward_flag = True
                        return True
                    # if self.time > 10:
                    #     self.reward_flag = True
                    #     self.done_flag = True
                    #     print('time')
                    #     return True

           
                def get_state(self):
                    return super(PerchingBixler,self).get_state()[0:14].T


                def get_normalized_obs(self, state=None):
                    if state is None:
                        state=self.get_state()

                    return state

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
                    self.time = 0.0
                    self.reward = 0.0
                    # self.target = np.array([30, -130])
                    self.sector_switch = 1

    return PerchingBixler