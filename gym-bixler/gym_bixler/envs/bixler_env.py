"""
Custom OpenAI Gym Enviornment for the Bixler2
performing a perched landing manoeuvre.
"""
import gym
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

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


        self.bixler.reset_scenario()
        self.state = self.bixler.get_state()
        
        state_list = self.state[0].tolist()
        state_list.insert(0, 0)
        self.state_array = []
        self.state_array.append(state_list)

        self.render_flag = False
    
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
        self.time = 0

        return self.bixler.get_normalized_state()

     
    def render(self, mode='file'):

        self.render_flag = True

        if mode == 'file':
            self._render_to_file()

    
    def _render_to_file(self):

        self.time += 0.1
        self.state = self.bixler.get_state()
        state_list = self.state[0].tolist()
        state_list.insert(0, self.time)
        self.state_array.append(state_list)
    

    def close(self):

        if self.render_flag == True:

            self.df = pd.DataFrame(self.state_array,columns = ['time', 'x','y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r', 'sweep', 'elev'])
            self.plot_data()
            plt.show()
            print(self.df.to_string())

            self.df.to_pickle('/tmp/gym/infer/state_data')
            self.df.to_csv('/tmp/gym/infer/state_data', index=False)

    def plot_data(self):

        fig = plt.figure()

        ax1 = fig.add_subplot(3,2,1)
        #pitch
        self.df['pitch'] = np.rad2deg(self.df['pitch'])
        self.df.plot(x = 'time', y = 'pitch',  ax = ax1, legend=False)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel(r'$\theta$ (deg)')
        ax1.grid()

        ax2 = fig.add_subplot(3,2,2)
        #pitch rate
        self.df['q'] = np.rad2deg(self.df['q'])
        self.df.plot(x = 'time', y = 'q',  ax = ax2, legend=False)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel(r'q (deg/s)')
        ax2.grid()

        ax3 = fig.add_subplot(3,2,3)
        #Sweep
        self.df.plot(x = 'time', y = 'sweep',  ax = ax3, legend=False)
        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel(r'Sweep (deg)')
        ax3.grid()

        ax4 = fig.add_subplot(3,2,4)
        #elev
        self.df.plot(x = 'time', y = 'elev',  ax = ax4, legend=False)
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel(r'Elevator (deg)')
        ax4.grid()

        ax5 = fig.add_subplot(3,2,5)
        #x vs y
        self.df['altitude'] = (self.df['z']*-1)
        self.df.plot(x = 'x', y = 'altitude',  ax = ax5, legend=False)
        ax5.set_xlabel("x position (m)")
        ax5.set_ylabel(r'Height (m)')
        ax5.grid()

        ax6 = fig.add_subplot(3,2,6)
        #airspeed
        self.df['airspeed'] = np.sqrt((self.df['u']**2)+(self.df['v']**2)+(self.df['w']**2))

        self.df.plot(x = 'time', y = 'airspeed',  ax = ax6, legend=False)
        ax6.set_xlabel("Time (seconds)")
        ax6.set_ylabel(r'Airspeed (m/s)')
        ax6.grid()






        

        

        
        
        




        
