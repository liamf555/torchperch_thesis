import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import os
import sys

path = sys.argv[1]

files = os.listdir(path)

fig = plt.figure()
lines = ["-","--","-.",":","-","--","-.",":"]

colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:purple', 'tab:gray', 'tab:purple']


for i, f in enumerate(files, 1):

    df = pd.read_csv(f'{path}{f}')

    ax1 = fig.add_subplot(3,2,1)
    #pitch
    df.plot(x = 't', y = 'theta', label = os.path.splitext(f)[0], ax = ax1, legend=False, color=colours[i])
    ax1.set_xlabel("Time (s)", fontsize=18)
    ax1.set_ylabel(r'$\theta$ (deg)', fontsize=18)
    ax1.tick_params(labelsize=12)
    ax1.grid()

    ax2 = fig.add_subplot(3,2,2)
    #pitch rate
    df.plot(x = 't', y = 'q',  ax = ax2, label = os.path.splitext(f)[0],  legend=True, color=colours[i])
    ax2.set_xlabel("Time (s)", fontsize=18)
    ax2.set_ylabel(r'q (deg/sec)', fontsize=18)
    ax2.tick_params(labelsize=12)
    ax2.grid()

    ax3 = fig.add_subplot(3,2,3)
    #Sweep
    df.plot(x = 't', y = 'sweep',  ax = ax3, label = os.path.splitext(f)[0],  legend=True, color=colours[i])
    ax3.set_xlabel("Time (s)", fontsize=18)
    ax3.set_ylabel(r'Sweep (deg)', fontsize=18)
    ax3.tick_params(labelsize=12)
    ax3.grid()

    ax4 = fig.add_subplot(3,2,4)
    #elev
    df.plot(x = 't', y = 'elev',  ax = ax4, label = os.path.splitext(f)[0],  legend=True, color=colours[i])
    ax4.set_xlabel("Time (s)", fontsize=18)
    ax4.set_ylabel(r'Elevator (deg)', fontsize=18)
    ax4.tick_params(labelsize=12)
    ax4.grid()
    

    ax5 = fig.add_subplot(3,2,5)
    #x vs y
    df['altitude'] = (df['z']*-1)
    df.plot(x = 'x', y = 'altitude',  ax = ax5, label = os.path.splitext(f)[0],  legend=True, color=colours[i])
    ax5.set_xlabel("x-Position (m)", fontsize=18)
    ax5.set_ylabel(r'Height (m)', fontsize=18)
    ax5.tick_params(labelsize=12)
    ax5.grid()
    

    ax6 = fig.add_subplot(3,2,6)
    # airspeed
    # df.plot(x = 't', y = 'airspeed',  ax = ax6, label = os.path.splitext(f)[0],  legend=True, linestyle=lines[i])
    
    # ax6.set_ylabel(r'Airspeed (m/s)', fontsize=18)
    ax6.grid()

    df.plot(x = 't', y = 'u', ax = ax6, legend=True,  linestyle='-', label = os.path.splitext(f)[0], color=colours[i])
    df.plot(x = 't', y = 'w', ax = ax6, legend=True,  linestyle='--', label = os.path.splitext(f)[0], color=colours[i])

    custom_lines = [Line2D([0], [0], color='k', linestyle = '-',  lw=2),
                        Line2D([0], [0], color='k', linestyle = '--', lw=2)]


    ax6.legend(custom_lines, ['u', 'w'])
    # # self.df.plot(x = 't', y = 'airspeed', ax = ax6)
    # ax6.set_xlabel("Time(s)", fontsize=12)
    ax6.set_xlabel("Time (s)", fontsize=18)
    ax6.set_ylabel(r'Body Velocities (m/s)', fontsize=18)
    ax6.tick_params(labelsize=12)
    ax6.grid()
    # ax6.grid()






axis = [ax1, ax2, ax3, ax4, ax5]

for ax in axis:

    handles, labels = ax.get_legend_handles_labels()

# sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles))) 
    # """key = lambda item: int(item[0])"""

    ax.legend(handles, labels, prop={'size': 12})

plt.show()







