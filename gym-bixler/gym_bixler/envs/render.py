import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Rendermixin(object): 

    def save_data(self):

        self.df = pd.DataFrame(self.state_array,columns = ['time', 'x','y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r', 'sweep', 'elev'])

        self.df['pitch'] = np.rad2deg(self.df['pitch'])
        self.df['q'] = np.rad2deg(self.df['q'])
        self.df['altitude'] = (self.df['z']*-1)
        self.df['airspeed'] = np.sqrt((self.df['u']**2)+(self.df['v']**2)+(self.df['w']**2))

        print(f'Reward: {self.reward}')

        self.df.to_pickle('../data/output.pkl')
        self.df.to_csv('../data/output.csv', index=False)

    def plot_data(self):

        if self.plot_flag == True:
            fig = plt.figure()

            ax1 = fig.add_subplot(3,2,1)
            #pitch
            
            self.df.plot(x = 'time', y = 'pitch',  ax = ax1, legend=False)
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel(r'$\theta$ (deg)')
            ax1.grid()

            ax2 = fig.add_subplot(3,2,2)
            #pitch rate
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
            
            self.df.plot(x = 'x', y = 'altitude',  ax = ax5, legend=False)
            ax5.set_xlabel("x position (m)")
            ax5.set_ylabel(r'Height (m)')
            ax5.grid()
            

            ax6 = fig.add_subplot(3,2,6)
            #airspeed
            
            self.df.plot(x = 'time', y = 'airspeed',  ax = ax6, legend=False)
            ax6.set_xlabel("Time (seconds)")
            ax6.set_ylabel(r'Airspeed (m/s)')
            ax6.grid()

            plt.show()