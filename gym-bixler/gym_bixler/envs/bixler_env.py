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

from scenarios import perching

import logging
logger = logging.getLogger(__name__) 

class BixlerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.bixler = perching.wrap_class('sweep_elevator', [])
        
        self.action_space = spaces.Discrete(self.bixler.actions)

        self.observation_space = spaces.Box(low =0, high =1, shape = self.bixler.state_dims)

        self.state = None

        


    def _step(self, action):
        # peform action

        episode_over = False
        self.bixler.set_action(action)

        #bixler step function with timestep 1
        self.bixler.step(0.1)

        #get reward
        reward = self.get_reward(episode_over)

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


        return self.state, reward, episode_over, info
        


    def _reset(self):

        self.bixler.reset_scenario()

        return self.bixler.get_normalized_state()


     
    def _render(self, mode='human', close=False):
      print(f'Reward: {reward}')
        

    # def _take_action(self, action):
    #     self.bixler.set_action(action)

    def _get_reward(self, epsiode_over):
        #get_reward function from perching.py, removing pytorch functions
        if self.bixler.is_terminal():
            episode_over = True
            if self.bixler.bixler.is_out_of_bounds():
                return self.bixler.failReward
            cost_vector = np.array([1,0,1, 0,100,0, 10,0,10, 0,0,0, 0,0 ])
            cost = np.dot( np.squeeze(self.bixler.get_state()) ** 2, cost_vector ) / 2500
            return ((1 - cost) * 2) - 1
        return 0

        
