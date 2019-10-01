"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import numpy as np 

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

    def __init__(self, controller='sweep_elevator',
			scenario='perching',
			scenario_opts=''):

        self.scenario = check_scenario(scenario)
        self.controller = check_controller(controller)
		
        scenario_args = None
        if len(scenario_opts) is not 0:
            scenario_args = self.scenario.parser.parse_args(args.scenario_opts[0].split(' '))
        else:
            scenario_args = self.scenario.parser.parse_args([])

        self.bixler = self.scenario.wrap_class(self.controller, scenario_args)()

        self.action_space = gym.spaces.Discrete(self.scenario.actions)

        self.observation_space = gym.spaces.Box(low=0, high = 1, shape = (1, self.scenario.state_dims))

        self.reward_range = (0.0,1.0)

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

        #bixler step function with timestep 1
        self.bixler.step(0.1)
		
		#get observation
        obs = self.bixler.get_normalized_state()

        #get reward
        self.reward = self.bixler.get_reward()
		
        done = self.bixler.is_terminal()

        info = {}

        return obs, self.reward, done, info
        

    def reset(self):

        self.bixler.reset_scenario()
        self.time = 0

        return self.bixler.get_normalized_state()

     
    def render(self, mode):

        self.render_flag = True

        if mode == 'save_file':
            self.create_array

        if mode == 'plot':
            self.plot_flag = True
            self.create_array()

    
    def create_array(self):

        self.time += 0.1
        self.state = self.bixler.get_state()
        state_list = self.state[0].tolist()
        state_list.insert(0, self.time)
        self.state_array.append(state_list)
        
    def close(self):

        if self.render_flag == True:
            Rendermixin.save_data(self)
        
        if self.plot_flag == True:
            Rendermixin.plot_data(self)



        