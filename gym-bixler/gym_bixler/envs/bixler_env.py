"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import numpy as np 

import controllers
import scenarios

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

class BixlerEnv(gym.Env):
    metadata = {'render.modes': ['file', 'none']}

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

        self.state = self.bixler.get_state[1]

    def step(self, action):
        # peform action

        self.bixler.set_action(action)

        #bixler step function with timestep 1
        self.bixler.step(0.1)
		
		#get observation
        obs = self.bixler.get_normalized_state()

        #get reward
        reward = self.bixler.get_reward()
		
        done = self.bixler.is_terminal()

        info = {}

        return obs, reward, done, info
        


    def reset(self):

        self.bixler.reset_scenario()

        return self.bixler.get_normalized_state()

     
    def render(self, mode='file', **kwargs):

        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))
    
     
    def state_data(self):

        self.time += 0.1
        self.state = self.bixler.get_state[1]

    def _render_to_file(self, filename='/tmp/gym/render.txt'):

        
        with open file (filename, 'a+') as file:

            file.write(f'Time: {self.time}')
            file.write(f'Altitude: {-self.state}')

        file.close()
        




        
