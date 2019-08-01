"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""

import os, subprocess, time, signal
import gym
from gym import error, spaces, logger
from gym import utils
from gym.utils import seeding
import numpy as np 
import argparse

import controllers
import scenarios

import logging
logger = logging.getLogger(__name__) 

class BixlerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        parser = argparse.ArgumentParser(description='Q-Learning for UAV manoeuvres in PyTorch')
        parser.add_argument('--controller', type=self.check_controller, default='sweep_elevator')
        parser.add_argument('--scenario', type=self.check_scenario, default='perching')
        parser.add_argument('--scenario-opts', nargs=1, type=str, default='')
        parser.add_argument('--logfile', type=argparse.FileType('w'), default='learning_log.txt')
        parser.add_argument('--networks', type=self.check_folder, default='networks' )
        parser.add_argument('--no-stdout', action='store_false', dest='use_stdout', default=True)
        args = parser.parse_args()

        
        scenario = args.scenario

        scenario_args = None
        if len(args.scenario_opts) is not 0:
            scenario_args = scenario.parser.parse_args(args.scenario_opts[0].split(' '))
        else:
            scenario_args = scenario.parser.parse_args([])

        self.bixler = scenario.wrap_class(args.controller, scenario_args)()

        self.action_space = spaces.Discrete(49)

        self.reward = 0

        self.observation_space = spaces.Box(low=0, high = 1, shape = self.bixler.get_normalized_state().shape)

        self.state = None

        


    def step(self, action):
        # peform action

        self.episode_over = False

        self.bixler.set_action(action)

        #bixler step function with timestep 1
        self.bixler.step(0.1)

        #get reward
        self.reward = self.get_reward()

        #get observation

        self.state = self.bixler.get_normalized_state()
        


        #    ob (object) :
        #         an environment-specific object representing your observation of
        #         the environment.
        #     reward (float) :
        #         amount of reward achieved by the previous action. The scale
        #         varies between environments, but the goal is always to increase
        #         your total reward.
        #     episode_over (bool) :
        #         whether it's time to reset the environment again. Most (but not
        #         all) tasks are divided up into well-defined episodes, and done
        #         being True indicates the episode has terminated. (For example,
        #         perhaps the pole tipped too far, or you lost your last life.)
        #     info (dict) :
        #          diagnostic information useful for debugging. It can sometimes
        #          be useful for learning (for example, it might contain the raw
        #          probabilities behind the environment's last state change).
        #          However, official evaluations of your agent are not allowed to
        #          use this for learning.

        info = {}


        return self.state, self.reward, self.episode_over, info
        


    def reset(self):

        self.bixler.reset_scenario()

        return self.bixler.get_normalized_state()


     
    def render(self, mode='human'):
      print(f'State: {self.bixler.get_state()}')
        


    def get_reward(self):
        #get_reward function from perching.py, removing pytorch functions
        if self.bixler.is_terminal():
            self.episode_over = True
            if self.bixler.is_out_of_bounds():
                return -1 #failreward
            cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0 ])
            cost = np.dot( np.squeeze(self.bixler.get_state()) ** 2, cost_vector ) / 2500
            return ((1 - cost) * 2) - 1
        return 0

    def check_controller(self, controller_name):
        if hasattr(controllers,controller_name):
            return getattr(controllers,controller_name)
        else:
            msg = "Could not find controller {}".format(controller_name)
            raise argparse.ArgumentTypeError(msg)

    def check_scenario(self, scenario_name):
        if hasattr(scenarios,scenario_name):
            return getattr(scenarios,scenario_name)
        else:
            msg = "Could not find scenario {}".format(scenario_name)
            raise argparse.ArgumentTypeError(msg)

    def check_folder(self, folder_name):
        if os.path.exists(folder_name):
            if os.path.isdir(folder_name):
                return folder_name
            else:
                msg = "File {} exists and is not a directory".format(folder_name)
                raise argparse.ArgumentTypeError(msg)
        os.mkdir(folder_name)
        return folder_name

        
