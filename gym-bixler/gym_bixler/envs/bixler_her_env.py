"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import json
import numpy as np 
from gym.utils import seeding
from gym import GoalEnv, spaces
import controllers
import scenarios
import wandb
import pprint as pp

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

class BixlerHEREnv(GoalEnv, Rendermixin):

    def __init__(self, parameters):
        

        self.scenario = check_scenario(parameters.get("scenario"))
        self.controller = check_controller(parameters.get("controller"))

        self.bixler = self.scenario.wrap_class(self.controller, parameters)()

        if parameters.get("controller") == "sweep_elevator_cont_rate": 
            self.action_space = gym.spaces.Box(low = np.array([-1, -1]),
                                            high = np.array([1, 1]), dtype = np.float16)

        else:
            self.action_space = gym.spaces.Discrete(self.scenario.actions)

        self.desired_goal = np.array([0, 0, 0, 0, 0])

        obs = self._get_obs()

        self.observation_space = spaces.Dict(dict(
            desired_goal =spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal = spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation = spaces.Box(-np.inf, np.inf, shape = obs['observation'].shape , dtype = 'float32'),
        ))

        self.reset()

        self.state = self.bixler.get_state()
        self.state = np.concatenate((self.state, self.bixler.velocity_e.T), axis = 1)
        state_list = self.state[0].tolist()
        state_list.insert(0, 0)
        state_list.append(self.bixler.alpha)
        state_list.append(self.bixler.airspeed)
        self.state_array = []
        self.state_array.append(state_list)

        self.render_flag = False
        self.plot_flag = False

        
    def step(self, action):
        # peform action

        self.bixler.set_action(action)

        #bixler step function with timestep 0.1
        try:
            self.bixler.step(0.1)
        except FloatingPointError:
            done = True
            obs = self._get_obs()
            self.reward = -1
            info = {}
       	else:
		    #get observation
            obs = self._get_obs()

            #get reward
            self.reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
            
            done = self.bixler.is_terminal()

            info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }

        return obs, self.reward, done, info
        

    def reset(self):

        self.bixler.reset_scenario()
        self.time = 0

        return self._get_obs()



     
    def render(self, mode):

        self.render_flag = True

        if mode == 'save':
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
        state_list.append(self.bixler.airspeed)
        self.state_array.append(state_list)
        
        
    def close(self, path, reward):
        if self.render_flag:
            Rendermixin.save_data(self, path, reward)

    def compute_reward(self, achieved_goal, desired_goal, _info):

        # Deceptive reward: it is positive only when the goal is achieved
        target_state = np.array([10, 0.1, 0.35, 8, 5], dtype='float32')

        d = np.abs(achieved_goal - desired_goal)

        if (achieved_goal != target_state).all():
            return 0 if (d < target_state).all() else -1 

    def _get_obs(self):

        obs = np.squeeze(self.bixler.get_normalized_obs())
        
        achieved_goal = np.float64(np.squeeze(np.delete(self.bixler.get_state(), [1, 3, 5, 7, 9, 10, 11, 12, 13], axis=1)))


        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.desired_goal,
        }

    def _is_success(self, achieved_goal, desired_goal):
        target_state = np.array([10, 0.1, 0.35, 10, 10], dtype='float32')
        d = np.abs(achieved_goal - desired_goal)

        wandb.log({'delta': d})

        return (d < target_state).all()