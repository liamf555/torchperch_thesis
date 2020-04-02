import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Rendermixin(object): 

    def save_data(self):

        self.df = pd.DataFrame(self.state_array,columns = ['time', 'x','y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r', 'sweep', 'elev', 'vn', 've', 'vd', 'alpha'])

        self.df['roll'] = np.rad2deg(self.df['roll'])
        self.df['pitch'] = np.rad2deg(self.df['pitch'])
        self.df['yaw'] = np.rad2deg(self.df['yaw'])
        self.df['q'] = np.rad2deg(self.df['q'])
        self.df['altitude'] = (self.df['z']*-1)
        
        self.df['airspeed'] = np.sqrt((self.df['u']**2)+(self.df['v']**2)+(self.df['w']**2))

        # print(f'Reward: {self.reward}')

        self.df.to_pickle('../data/output.pkl')
        self.df.to_csv('../data/output.csv', index=False)

    def plot_data(self):

        if self.plot_flag == True:
            fig = plt.figure()

            ax1 = fig.add_subplot(3,2,1)
            #pitch
            
            self.df.plot(x = 'time', y = 'pitch', color = 'k',  ax = ax1, legend=True, label = r'$\theta$')

            self.df.plot(x = 'time', y = 'alpha', color = 'orange',  ax = ax1, legend=True, label = 'AoA')
            ax1.set_xlabel("Time (seconds)", fontsize=18)
            ax1.set_ylabel("Angle (deg)", fontsize=18)
            ax1.grid()



            ax2 = fig.add_subplot(3,2,2)
            #pitch rate
            self.df.plot(x = 'time', y = 'q', color = 'k',  ax = ax2, legend=False)
            ax2.set_xlabel("Time (seconds)", fontsize=18)
            ax2.set_ylabel(r'q (deg/sec)', fontsize=18)
            ax2.grid()

            ax3 = fig.add_subplot(3,2,3)
            #Sweep
            self.df.plot(x = 'time', y = 'sweep', color = 'k',  ax = ax3, legend=False)
            ax3.set_xlabel("Time (seconds)", fontsize=18)
            ax3.set_ylabel(r'Sweep (deg)', fontsize=18)
            ax3.grid()

            ax4 = fig.add_subplot(3,2,4)
            #elev
            self.df.plot(x = 'time', y = 'elev', color = 'k',  ax = ax4, legend=False)
            ax4.set_xlabel("Time (seconds)", fontsize=18)
            ax4.set_ylabel(r'Elevator (deg)', fontsize=18)
            ax4.grid()
            

            ax5 = fig.add_subplot(3,2,5)
            #x vs y
            
            self.df.plot(x = 'x', y = 'altitude', color = 'k',  ax = ax5, legend=False)
            ax5.set_xlabel("x-Position (m)", fontsize=18)
            ax5.set_ylabel('Height (m)', fontsize=18)
            ax5.grid()
            

            ax6 = fig.add_subplot(3,2,6)

            # #airspeed
            
            # self.df.plot(x = 'time', y = 'airspeed',  ax = ax6, legend=False)
            # ax6.set_xlabel("Time (seconds)")
            # ax6.set_ylabel(r'Airspeed (m/s)')
            # ax6.grid()

            # body velocities

            self.df.plot(x = 'time', y = 'u', color = 'b',  ax = ax6, legend=True)
            self.df.plot(x = 'time', y = 'w', color = 'r',  ax = ax6, legend=True)
            ax6.set_xlabel("Time (seconds)", fontsize=18)
            ax6.set_ylabel(r'Body Velocities (m/s)', fontsize=18)
            ax6.grid()

            plt.show()