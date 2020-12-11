import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import plotly
# pd.options.plotting.backend = "plotly"
import plotly.tools as tls
import warnings
warnings.filterwarnings("ignore")

class PiParse:

    def __init__(self):

        self.ml_ep = False
        self.states = []
        self.ep_dict = {}
        self.state_dict = {}
        self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:gray', 'tab:purple']
        
    def get_pos(self, line):
        if "r_a" in line:
            line_list = (line.split("r_a: [[",1)[1].replace(']]', '').strip().split(" "))
            line_list = [float(x) for x in line_list if x != '']
            del line_list[1]
            self.pos = line_list


    def get_control(self, line):
        if 'sweep' in line:
            line_list = line.split(',')
            line_list = [float(i.split(": [", 1)[1].replace(']','').strip()) for i in line_list]
            self.cont = line_list

    def get_velocities(self, line):
        if 'u:' in line:
            line_list = line.split(',')
            line_list = [float(i.split(": [", 1)[1].replace(']','').strip()) for i in line_list]
            self.vel = line_list
            # print(self.vel)

    def get_pitch(self, line):
        if 'pitch' in line:
            line_list = line.split(',')
            line_list[0] = float(line_list[0].split(':[',1)[1].replace(']', '').strip())
            line_list[1] = float(line_list[1].split(': [',1)[1].replace(']', '').strip())
            self.pitch = line_list

    def get_states(self, line):
        pos = self.get_pos(line)
        cont = self.get_control(line)
        vel = self.get_velocities(line)
        pitch = self.get_pitch(line)

        if 'pitch' in line:
            self.states.append(self.pos + self.vel + self.pitch + self.cont)

    def make_fig(self):

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2)

    def plotter(self, dict_key):
    #   self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2)
      for i, dataframe in enumerate(self.state_dict[dict_key].values()):
        self.plot(dataframe, self.colours[i])
      # self.fig.savefig('graph.pdf')
    #   plotly_fig = tls.mpl_to_plotly(self.fig)
    #   plotly_fig.write_image("fig1.pdf")
    #   plotly.offline.iplot(plotly_fig)

    def plot(self, dataframe, colour):
      
        #Pitch(deg)
        dataframe['pitch'] = np.rad2deg(dataframe['pitch'])
        dataframe.plot(x = 't', y = 'pitch', ax = self.ax1, color = colour, legend=False, label = r'$\theta$')

        # self.df.plot(x = 't', y = 'alpha', color = 'orange',  ax = ax1, legend=True, label = 'AoA')
        self.ax1.set_xlabel("Time (s)", fontsize=18)
        self.ax1.set_ylabel(r'$\theta$ (deg)', fontsize=18)
        self.ax1.tick_params(labelsize=12)
        self.ax1.grid()

        #Pitch(deg) rate
        dataframe['q'] = np.rad2deg(dataframe['q'])
        dataframe.plot(x = 't', y = 'q', color = colour,  ax = self.ax2, legend=False)
        self.ax2.set_xlabel("Time (s)", fontsize=18)
        self.ax2.set_ylabel(r'q (deg/sec)', fontsize=18)
        self.ax2.tick_params(labelsize=12)
        self.ax2.grid()

        #Sweep
        dataframe.plot(x = 't', y = 'sweep', color = colour, ax = self.ax3, legend=False)
        self.ax3.set_xlabel("Time (s)", fontsize=18)
        self.ax3.set_ylabel(r'Sweep (deg)', fontsize=18)
        self.ax3.tick_params(labelsize=12)
        self.ax3.grid()

        #elev
        dataframe.plot(x = 't', y = 'elev', color = colour, ax = self.ax4, legend=False)
        self.ax4.set_xlabel("Time (s)", fontsize=18)
        self.ax4.set_ylabel(r'Elevator (deg)', fontsize=18)
        self.ax4.tick_params(labelsize=12)
        self.ax4.grid()

        #x vs y
        # ax5.invert_yaxis()
        dataframe['height'] = dataframe['z']*-1

        dataframe.plot(x = 'x', y = 'height', color = colour,  ax = self.ax5, legend=False)
        self.ax5.set_xlabel("x-Position (m)", fontsize=18)
        self.ax5.set_ylabel('Height (m)', fontsize=18)
        self.ax5.set_xlim(-40, 5)
        self.ax5.tick_params(labelsize=12)
        # ax5.set_ylim(-6, 0)
        self.ax5.grid()


        # #airspeed
        
        # self.df.plot(x = 'Time(sec)', y = 'airspeed',  ax = ax6, legend=False)
        # ax6.set_xlabel("Time(sec)")
        # ax6.set_ylabel(r'Airspeed (m/s)')
        # ax6.grid()

        # body velocities

        dataframe.plot(x = 't', y = 'u', ax = self.ax6, color = colour,legend=False)
        dataframe.plot(x = 't', y = 'w', ax = self.ax6, color = colour, linestyle='--',legend=False)
        self.ax6.set_xlabel("Time (s)", fontsize=18)
        self.ax6.set_ylabel(r'Body Velocities (m/s)', fontsize=18)
        self.ax6.tick_params(labelsize=12)
        self.ax6.grid()


        custom_lines = [Line2D([0], [0], color='k', linestyle = '-',  lw=2),
                        Line2D([0], [0], color='k', linestyle = '--', lw=2)]


        self.ax6.legend(custom_lines, ['u', 'w'])

    def get_reward(self, dataframe):
        def gaussian(x, sig = 0.4, mu = 0):   
                    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)


        
        final_state = dataframe.tail(1)   

        obs = np.array([final_state.iloc[0]['x'], final_state.iloc[0]['z'], final_state.iloc[0]['pitch'], final_state.iloc[0]['u'] , final_state.iloc[0]['w']])
        target_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype='float64')
        bound = np.array([15, 5, np.deg2rad(20),10,10])
        cost = (target_state - obs)/bound
        cost = list(map(gaussian, cost))
        reward = np.prod(cost)
        return reward

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-dir', type=str, required=True)
    # parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    file_directory = Path(args.directory)

    pi_parser = PiParse()

    results = {}

for log in file_directory.glob('*.txt'): 

    ep_number = 0
    ml_ep = False
    with open(log, "r") as file:
            for line in file:
                if 'Episode start' in line:
                    # print('new_ep')
                    ml_ep = True
                elif 'Episode end' in line:
                    ml_ep = False
                    ep_number +=1
                    df = pd.DataFrame(pi_parser.states, columns=['x', 'z', 'u', 'w', 'pitch', 'q', 'sweep', 'elev'])
                    pi_parser.ep_dict[ep_number] = df
                    pi_parser.states = []
                if ml_ep:
                    pi_parser.get_states(line)
    t = 0
    ts = []
    rewards = []
    for i, dataframe in enumerate(pi_parser.ep_dict.values()):
        rows = dataframe.shape[0]
        for j in range(rows):
            ts.append(t)
            t += 0.1
        dataframe['t'] = ts
        ep_reward = pi_parser.get_reward(dataframe)
        rewards.append(ep_reward)
        t = 0
        ts = []
    
    results[log.stem] = rewards
    pi_parser.state_dict[log.stem] = pi_parser.ep_dict
    pi_parser.ep_dict = {}
reward_df = pd.DataFrame.from_dict(results, orient='index')
reward_df = reward_df.transpose()
reward_df.loc['Mean'] = reward_df.mean()
reward_df.loc['SD'] = reward_df.std()

cols=reward_df.columns.tolist()
cols.sort()
reward_df=reward_df[cols]
print(reward_df)


pi_parser.make_fig()
# pi_parser.plotter("wind_1_tail_1")
pi_parser.plotter("gust_1_head_1")
# pi_parser.plotter("wind_2_head_1")

# pi_parser.plotter("baseline_1_head_2")

plt.show()
    
        



  
  
                    



                
                
                    
                    





                


