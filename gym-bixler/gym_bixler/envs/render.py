import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import wandb


class Rendermixin(object):
    def save_data(self, path, reward):

        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        self.df = pd.DataFrame(self.state_array,columns = ['t', 'x','y', 'z', 'phi', 'theta', 'psi', 'u', 'v', 'w', 'p', 'q', 'r', 'sweep', 'elev', 'vn', 've', 'vd', 'alpha', 'airspeed'])

        self.state_array = []

        self.df['phi'] = np.rad2deg(self.df['phi'])
        self.df['theta'] = np.rad2deg(self.df['theta'])
        self.df['psi'] = np.rad2deg(self.df['psi'])
        self.df['p'] = np.rad2deg(self.df['p'])
        self.df['q'] = np.rad2deg(self.df['q'])
        self.df['r'] = np.rad2deg(self.df['r'])
        self.df['altitude'] = (self.df['z']*-1)

        # print(f'Reward: {self.reward}')

        self.df.astype('float32').dtypes

        self.df.drop(self.df.tail(1).index,inplace=True)
        self.df_vis = self.df[['t', 'phi', 'theta', 'psi', 'vn', 've', 'vd', 'x','y','z']]
        self.df_vis.columns = ['Time(s)', 'Roll(deg)', 'Pitch(deg)', 'Yaw(deg)', 'VN(m/s)', 'VE(m/s)', 'VD(m/s)', 'PN(m)','PE(m)', 'PD(m)']
        
    
        # self.df.to_pickle(path+'.pkl')
        self.df.to_csv(path +'.csv', index=False, float_format='%.5f')
        self.df_vis.to_csv(path +'_vis.csv', index=False, float_format='%.5f')
        
        fig = plt.figure(figsize=(12.8, 9.6))

        if 'f' in path:
            fig_title = re.sub(r'.*f', 'f', path)
        else:
            fig_title = re.sub(r'.*b', 'b', path)

        fig_title = fig_title + f': Reward = {reward:.4f}'

        fig.suptitle(f'{fig_title}', fontsize=16)

        ax1 = fig.add_subplot(3,2,1)
        #Pitch(deg)
        
        self.df.plot(x = 't', y = 'theta', color = 'k',  ax = ax1, legend=True, label = r'$\theta$')

        self.df.plot(x = 't', y = 'alpha', color = 'orange',  ax = ax1, legend=True, label = 'AoA')
        ax1.set_xlabel("Time(s)", fontsize=12)
        ax1.set_ylabel("Angle (deg)", fontsize=12)
        ax1.grid()


        ax2 = fig.add_subplot(3,2,2)
        #Pitch(deg) rate
        self.df.plot(x = 't', y = 'q', color = 'k',  ax = ax2, legend=False)
        ax2.set_xlabel("Time(s)", fontsize=12)
        ax2.set_ylabel(r'q (deg/sec)', fontsize=12)
        ax2.grid()

        ax3 = fig.add_subplot(3,2,3)
        #Sweep
        self.df.plot(x = 't', y = 'sweep', color = 'k',  ax = ax3, legend=False)
        ax3.set_xlabel("Time(s)", fontsize=12)
        ax3.set_ylabel(r'Sweep (deg)', fontsize=12)
        ax3.grid()

        ax4 = fig.add_subplot(3,2,4)
        #elev
        self.df.plot(x = 't', y = 'elev', color = 'k',  ax = ax4, legend=False)
        ax4.set_xlabel("Time(s)", fontsize=12)
        ax4.set_ylabel(r'Elevator (deg)', fontsize=12)
        ax4.grid()

        
        ax5 = fig.add_subplot(3,2,5)
        #x vs y
        
        self.df.plot(x = 'x', y = 'altitude', color = 'k',  ax = ax5, legend=False)
        ax5.set_xlabel("x-Position (m)", fontsize=12)
        ax5.set_ylabel('Height (m)', fontsize=12)
        ax5.grid()
        

        ax6 = fig.add_subplot(3,2,6)

        # #airspeed
        
        # self.df.plot(x = 'Time(sec)', y = 'airspeed',  ax = ax6, legend=False)
        # ax6.set_xlabel("Time(sec)")
        # ax6.set_ylabel(r'Airspeed (m/s)')
        # ax6.grid()

        # body velocities

        self.df.plot(x = 't', y = 'u', color = 'b',  ax = ax6, legend=True)
        self.df.plot(x = 't', y = 'w', color = 'r',  ax = ax6, legend=True)
        self.df.plot(x = 't', y = 'airspeed', ax = ax6)
        ax6.set_xlabel("Time(s)", fontsize=12)
        ax6.set_ylabel(r'Body Velocities (m/s)', fontsize=12)
        ax6.grid()
        
        # plt.savefig(path+'.png')
        
        # wandb.log({fig_title: plt})
        
        if self.plot_flag:
            plt.show()