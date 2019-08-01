import gym
import numpy as np

import controllers
import scenarios

# Check that requested controller exists
def check_controller(controller_name):
	if hasattr(controllers,controller_name):
		return getattr(controllers,controller_name)
	else:
		msg = "Could not find controller {}".format(controller_name)
		raise ValueError(msg)

# Check that requested scenario exists
def check_scenario(scenario_name):
	if hasattr(scenarios,scenario_name):
		return getattr(scenarios,scenario_name)
	else:
		msg = "Could not find scenario {}".format(scenario_name)
		raise ValueError(msg)

class RobsBixlerEnv(gym.Env):

	def __init__(self,
			controller='sweep_elevator',
			scenario='perching',
			scenario_opts=''):

		# Check we can find the scenario and controller
		self.scenario = check_scenario(scenario)
		self.controller = check_controller(controller)

		if len(scenario_opts) is not 0:
			scenario_args = self.scenario.parser.parse_args(scenario_opts[0].split(' '))
		else:
			scenario_args = self.scenario.parser.parse_args([])
		
		self.vehicle_model = self.scenario.wrap_class(self.controller,scenario_args)()

		self.metadata = {'render.modes':[]} # None?

		self.reward_range = (0.0,1.0)       # Depends on the scenario
		self.spec = None                    # What?

		self.action_space = gym.spaces.Discrete(self.scenario.actions) # Depends on scenario
		self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,self.scenario.state_dims) ) # Depends on scenario

	def step(self,action):
		# Advance dynamics
		self.vehicle_model.set_action(action)
		self.vehicle_model.step(0.1)
	
		# Collect state & reward	
		observation = self.vehicle_model.get_normalized_state()
		reward = self.vehicle_model.get_reward()
		done = self.vehicle_model.is_terminal()
		info = {}
		return observation, reward, done, info

	def reset(self):
		self.vehicle_model.reset_scenario()
		return self.vehicle_model.get_normalized_state()

	def render(self,mode):
		# If we're having any render mode, just print the state?
		# Let it be someone else's problem for now:
		super(RealBixlerEnv,self).render(mode=mode)
