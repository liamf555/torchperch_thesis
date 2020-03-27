"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import json
import numpy as np 
from gym.utils import seeding
import controllers
import scenarios

from gym_bixler.envs.render import Rendermixin

def check_controller(controller_name):
	if hasattr(controllers,controller_name):
		return getattr(controllers,controller_name)
	else:
		msg = "Could not find controller {}".format(controller_name)
		raise ValueError(msg)

def check_scenario(scenario_name):
	if hasattr(scenarios,scenario_name):
		return getattr(scenarios,scenario_name)
	else:
		msg = "Could not find scenario {}".format(scenario_name)
		raise ValueError(msg)

class BixlerEnv(Rendermixin, gym.Env):
    metadata = {'render.modes': ['save_file', 'plot', 'none']}

    def __init__(self, parameters):

        self.scenario = check_scenario(parameters.get("scenario"))
        self.controller = check_controller(parameters.get("controller"))

        self.bixler = self.scenario.wrap_class(self.controller, parameters)()

        if parameters.get("controller") == "sweep_elevator":
            self.action_space = gym.spaces.Discrete(self.scenario.actions)

        if parameters.get("controller") == "sweep_elevator_cont_rate": 
            self.action_space = gym.spaces.Box(low = np.array([-1, -1]),
                                            high = np.array([1, 1]), dtype = np.float16)

        self.observation_space = gym.spaces.Box(low=0, high = 1, shape = (1, self.scenario.state_dims), dtype = np.float64)

        self.bixler.reset_scenario()
        self.state = self.bixler.get_state()
        
        state_list = self.state[0].tolist()
        state_list.insert(0, 0)

        self.state_array = []
        self.state_array.append(state_list)

        self.render_flag = False
        self.plot_flag = False

        
    def step(self, action):
        # peform action

        self.bixler.set_action(action)

        #bixler step function with timestep 0.1
        self.bixler.step(0.1)
     	
		#get observation
        obs = self.bixler.get_normalized_obs()

       
        #get reward
        self.reward = self.bixler.get_reward()
		
        done = self.bixler.is_terminal()

        info = {}

        return obs, self.reward, done, info
        

    def reset(self):

        self.bixler.reset_scenario()
        self.time = 0

        return self.bixler.get_normalized_obs()



     
    def render(self, mode):

        self.render_flag = True

        if mode == 'save_file':
            self.create_array()

        if mode == 'plot':
            self.plot_flag = True
            self.create_array()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        self.bixler.seed(seed)

        return [seed]


    def create_array(self):

        self.time += 0.1

        self.state = self.bixler.get_state()
        self.state = np.concatenate((self.state, self.bixler.velocity_e.T), axis = 1)
        state_list = self.state[0].tolist()
        state_list.insert(0, self.time)
        state_list.append(self.bixler.alpha)

        self.state_array.append(state_list)
        
    def close(self):

        if self.render_flag:
            Rendermixin.save_data(self)
        
        if self.plot_flag:
            Rendermixin.plot_data(self)



        