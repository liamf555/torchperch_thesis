"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import json
import numpy as np
import controllers
import scenarios
try:
    import wandb
except ImportError:
    pass

from gym_bixler.envs.render import Rendermixin


def check_controller(controller_name):
    if hasattr(controllers, controller_name):
        return getattr(controllers, controller_name)
    else:
        msg = "Could not find controller {}".format(controller_name)
        raise ValueError(msg)


def check_scenario(scenario_name):
    if hasattr(scenarios, scenario_name):
        return getattr(scenarios, scenario_name)
    else:
        msg = "Could not find scenario {}".format(scenario_name)
        raise ValueError(msg)


class BixlerEnv(Rendermixin, gym.Env):
    metadata = {'render.modes': ['save', 'plot', 'none']}

    def __init__(self, parameters):

        self.scenario = check_scenario(parameters.get("scenario"))
        self.controller = check_controller(parameters.get("controller"))
        self.parameters = parameters

        # if self.parameters.get("turbulence"):

        # self.parameters["dryden_gusts"] =

        self.bixler = self.scenario.wrap_class(
            self.controller, self.parameters)()

        # self.seed(self.parameters.get("seed"))
        # self.seed()

        if self.parameters.get("controller") == "sweep_elevator_cont_rate":
            self.action_space = gym.spaces.Box(low=np.array([-1, -1]),
                                               high=np.array([1, 1]), dtype=np.float32)
        elif self.parameters.get("controller") == "elevator_cont":
            self.action_space = gym.spaces.Box(low=np.array([-1]),
                                               high=np.array([1]), dtype=np.float32)
        elif self.parameters.get("controller") == "throttle_delay":
            self.action_space = gym.spaces.MultiDiscrete([7, 7, 2])
        elif self.parameters.get("controller") == "throttle_delay_cont":
            self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]),
                                               high=np.array([1, 1, 1]), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(self.scenario.actions)

        self.observation_space = gym.spaces.Dict(
            spaces={
                "vec": gym.spaces.Box(-np.inf, np.inf, (self.scenario.state_dims,), dtype=np.float32),
                "throttle": gym.spaces.Discrete(2),
            }
        )

        # self.observation_space=gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(1, self.scenario.state_dims), dtype=np.float32)

        self.bixler.reset_scenario()

        # self.state = self.bixler.get_state()
        # self.state = np.concatenate((self.state, self.bixler.velocity_e.T), axis = 1)
        # state_list = self.state[0].tolist()
        # state_list.insert(0, 0)
        # state_list.append(self.bixler.alpha)
        # state_list.append(self.bixler.airspeed)
        # self.state_array = []
        # self.state_array.append(state_list)

        self.training = parameters.get("training")
        self.render_flag_entry = True

        self.plot_flag = False
        self.save_flag = False

        # self.time = 0.0

    def step(self, action):
        # peform action

        self.bixler.set_action(action)

        # print(action)

        # bixler step function with timestep 0.1
        try:
            self.bixler.step(0.1)
        except FloatingPointError:
            done = True
            obs = self.bixler.get_normalized_obs()
            self.reward = 0.0
            info = {}
        else:
            # get observation
            obs = self.bixler.get_normalized_obs()

            self.reward = self.bixler.get_reward()

            done = self.bixler.is_terminal()

            info = {}

        return obs, self.reward, done, info

    def reset(self):

        self.bixler.reset_scenario()
        # print('new ep')

        if not self.training:
            self.time = 0.0
            # print(self.bixler.wind)
            if self.render_flag_entry:
                self.state = self.bixler.get_state()

                self.state = np.concatenate(
                    (self.state, self.bixler.velocity_e.T), axis=1)
                state_list = self.state[0].tolist()
                state_list.insert(0, 0)
                state_list.append(self.bixler.alpha)
                state_list.append(self.bixler.airspeed)
                state_list.append(self.bixler.throttle)
                self.state_array = []
                self.state_array.append(state_list)
                self.render_flag_entry = False

        # print(self.bixler.get_normalized_obs())

        return self.bixler.get_normalized_obs()

    def render(self, mode):

        if mode == 'save':
            self.create_array()
            self.save_flag = True

        if mode == 'plot':
            self.plot_flag = True
            self.save_flag = True
            self.create_array()

        if mode == None:
            pass

    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     self.bixler.seed(seed)

    #     return [seed]

    def create_array(self):

        self.time += 0.1
        self.state = self.bixler.get_state()
        self.state = np.concatenate(
            (self.state, self.bixler.velocity_e.T), axis=1)
        state_list = self.state[0].tolist()
        state_list.insert(0, self.time)
        state_list.append(self.bixler.alpha)
        state_list.append(self.bixler.airspeed)
        state_list.append(self.bixler.throttle)
        self.state_array.append(state_list)

    # def close(self, path, reward):
    #     if self.render_flag:
    #         Rendermixin.save_data(self, path, reward)

    def save_plots(self, path, reward):
        if self.save_flag:
            Rendermixin.save_data(self, path, reward)
