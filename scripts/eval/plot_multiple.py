import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

path = sys.argv[1]

files = os.listdir(path)

dataframes = {}

fig = plt.figure()
lines = ["-","--","-.",":","-","--","-.",":"]


for i, f in enumerate(files, 1):

    df = pd.read_pickle(f'{path}{f}')

    ax1 = fig.add_subplot(3,2,1)
    #pitch
    df.plot(x = 'time', y = 'pitch', label = os.path.splitext(f)[0],  ax = ax1, legend=True, linestyle = lines[i])
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel(r'$\theta$ (deg)')
    ax1.grid()

    ax2 = fig.add_subplot(3,2,2)
    #pitch rate
    df.plot(x = 'time', y = 'q',  ax = ax2, label = os.path.splitext(f)[0],  legend=True, linestyle=lines[i])
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel(r'q (deg/s)')
    ax2.grid()

    ax3 = fig.add_subplot(3,2,3)
    #Sweep
    df.plot(x = 'time', y = 'sweep',  ax = ax3, label = os.path.splitext(f)[0],  legend=True, linestyle=lines[i])
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel(r'Sweep (deg)')
    ax3.grid()

    ax4 = fig.add_subplot(3,2,4)
    #elev
    df.plot(x = 'time', y = 'elev',  ax = ax4, label = os.path.splitext(f)[0],  legend=True, linestyle=lines[i])
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel(r'Elevator (deg)')
    ax4.grid()
    

    ax5 = fig.add_subplot(3,2,5)
    #x vs y
    df['altitude'] = (df['z']*-1)
    df.plot(x = 'x', y = 'altitude',  ax = ax5, label = os.path.splitext(f)[0],  legend=True, linestyle=lines[i])
    ax5.set_xlabel("x position (m)")
    ax5.set_ylabel(r'Height (m)')
    ax5.grid()
    

    ax6 = fig.add_subplot(3,2,6)
    #airspeed
    df.plot(x = 'time', y = 'airspeed',  ax = ax6, label = os.path.splitext(f)[0],  legend=True, linestyle=lines[i])
    ax6.set_xlabel("Time (seconds)")
    ax6.set_ylabel(r'Airspeed (m/s)')
    ax6.grid()


ax1.legend()
plt.show()







