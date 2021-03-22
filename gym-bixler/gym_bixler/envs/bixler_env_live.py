"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import json
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

parameters = {
    "scenario": "perching_long",
    "controller": "sweep_elevator",
    "noise": 0.0,
    "wind_mode": "None",
    "wind_params": [0, 0, 0],
}

class BixlerEnvLive(gym.Env):

    def __init__(self, parameters):

        
        self.scenario = check_scenario(parameters.get("scenario"))
        self.controller = check_controller(parameters.get("controller"))
        
        self.bixler = self.scenario.wrap_class(self.controller, parameters)()

        if parameters.get("controller") == "sweep_elevator_cont_rate":
            self.action_space = gym.spaces.Box(low = np.array([-1, -1]),
                                            high = np.array([1, 1]), dtype = np.float16)
        elif parameters.get("controller") == "elevator_cont":
            self.action_space = gym.spaces.Box(low = np.array([-1]),
                                            high = np.array([1]), dtype = np.float16)
        else:
            self.action_space = gym.spaces.Discrete(self.scenario.actions)

        self.observation_space = gym.spaces.Box(low=-np.inf, high = np.inf, shape = (1, self.scenario.state_dims), dtype = np.float64)

        self.bixler.reset_scenario()

    def step(self, action):
        # peform action

        self.bixler.set_action(action)

        #bixler step function with timestep 0.1
        try:
            self.bixler.step(0.1)
        except FloatingPointError:
            done = True
            obs = self.bixler.get_normalized_obs()
            self.reward = 0.0
            info = {}
       	else:
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
        # self.dryden.update

        return self.bixler.get_normalized_obs()

    def render(self, mode):
        pass

    def set_bixler_action(self, action):
        self.bixler.set_action(action)  
