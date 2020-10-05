import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

class PiParse:

    def __init__(self):

        self.ml_ep = False
        self.states = []
        self.ep_dict = {}
        
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

    def plot(self, dataframe, colour):
    
        #Pitch(deg)
        dataframe['pitch'] = np.rad2deg(dataframe['pitch'])
        dataframe.plot(x = 't', y = 'pitch', ax = ax1, color = colour, legend=False, label = r'$\theta$')

        # self.df.plot(x = 't', y = 'alpha', color = 'orange',  ax = ax1, legend=True, label = 'AoA')
        ax1.set_xlabel("Time(s)", fontsize=12)
        ax1.set_ylabel("Pitch (deg)", fontsize=12)
        ax1.grid()

        #Pitch(deg) rate
        dataframe['q'] = np.rad2deg(dataframe['q'])
        dataframe.plot(x = 't', y = 'q', color = colour,  ax = ax2, legend=False)
        ax2.set_xlabel("Time(s)", fontsize=12)
        ax2.set_ylabel(r'q (deg/sec)', fontsize=12)
        ax2.grid()

        #Sweep
        dataframe.plot(x = 't', y = 'sweep', color = colour, ax = ax3, legend=False)
        ax3.set_xlabel("Time(s)", fontsize=12)
        ax3.set_ylabel(r'Sweep (deg)', fontsize=12)
        ax3.grid()

        #elev
        dataframe.plot(x = 't', y = 'elev', color = colour, ax = ax4, legend=False)
        ax4.set_xlabel("Time(s)", fontsize=12)
        ax4.set_ylabel(r'Elevator (deg)', fontsize=12)
        ax4.grid()

        #x vs y
        # ax5.invert_yaxis()
        dataframe['height'] = dataframe['z']*-1

        dataframe.plot(x = 'x', y = 'height', color = colour,  ax = ax5, legend=False)
        ax5.set_xlabel("x-Position (m)", fontsize=12)
        ax5.set_ylabel('Height (m)', fontsize=12)
        ax5.set_xlim(-40, 5)
        # ax5.set_ylim(-6, 0)
        ax5.grid()


        # #airspeed
        
        # self.df.plot(x = 'Time(sec)', y = 'airspeed',  ax = ax6, legend=False)
        # ax6.set_xlabel("Time(sec)")
        # ax6.set_ylabel(r'Airspeed (m/s)')
        # ax6.grid()

        # body velocities

        dataframe.plot(x = 't', y = 'u', ax = ax6, color = colour,legend=False)
        dataframe.plot(x = 't', y = 'w', ax = ax6, color = colour, linestyle='--',legend=False)
        ax6.set_xlabel("Time(s)", fontsize=12)
        ax6.set_ylabel(r'Body Velocities (m/s)', fontsize=12)
        ax6.grid()

        custom_lines = [Line2D([0], [0], color='k', linestyle = '-',  lw=2),
                        Line2D([0], [0], color='k', linestyle = '--', lw=2)]


        ax6.legend(custom_lines, ['u', 'w'])

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
    parser.add_argument('--plot', action='store_true')
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
                        ml_ep = True
                        pi_parser.states = []
                    elif 'Episode end' in line:
                        ml_ep = False
                        ep_number +=1
                        # print(pi_parser.states)
                        df = pd.DataFrame(pi_parser.states, columns=['x', 'z', 'u', 'w', 'pitch', 'q', 'sweep', 'elev'])
                        pi_parser.ep_dict[ep_number] = df
                        pi_parser.states = []
                    if ml_ep:
                        pi_parser.get_states(line)
        t = 0
        ts = []
        colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:gray', 'tab:purple']
        rewards = [] 
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        for i, dataframe in enumerate(pi_parser.ep_dict.values()):
            rows = dataframe.shape[0]
            for j in range(rows):
                ts.append(t)
                t += 0.1
            dataframe['t'] = ts
            colour = colours[i]
            # print(dataframe)
            ep_reward = pi_parser.get_reward(dataframe)
            rewards.append(ep_reward)
            pi_parser.plot(dataframe, colour)
            t = 0
            ts = []
            # print(f"Episode: {i} Reward: {ep_reward}")
        results[log.stem] = rewards
        if args.plot:
            plt.show()
    final_df = pd.DataFrame.from_dict(results, orient='index')
    final_df = final_df.transpose()
    final_df.loc['Mean'] = final_df.mean()
    final_df.loc['SD'] = final_df.std()
    print(final_df)
    
        



  
  
                    



                
                
                    
                    





                


