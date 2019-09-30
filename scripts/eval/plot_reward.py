import gym
import gym_bixler

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors

import argparse
import sys, os

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import DQN
from stable_baselines import PPO1
from stable_baselines import ACKTR
from stable_baselines import A2C
from stable_baselines import TRPO


path = sys.argv[1]

files = os.listdir(path)


fig1, ax1 = plt.subplots(figsize=(15,10))
fig2, ax2 = plt.subplots(figsize=(15,10))
fig3, ax3 = plt.subplots(figsize=(15,10))
fig4, ax4 = plt.subplots(figsize=(15,10))
fig5, ax5 = plt.subplots(figsize=(15,10))
fig6, ax6 = plt.subplots(figsize=(15,10))
fig7, ax7 = plt.subplots(figsize=(15,10))
fig8, ax8 = plt.subplots(figsize=(15,10))
fig9, ax9 = plt.subplots(figsize=(15,10))
fig10, ax10 = plt.subplots(figsize=(15,10))
fig11, ax11 = plt.subplots(figsize=(15,10))
fig12, ax12 = plt.subplots(figsize=(15,10))
fig13, ax13 = plt.subplots(figsize=(15,10))
fig14, ax14 = plt.subplots(figsize=(15,10))
fig15, ax15 = plt.subplots(figsize=(15,10))
fig16, ax16 = plt.subplots(figsize=(15,10))
fig17, ax17 = plt.subplots(figsize=(15,10))
fig18, ax18 = plt.subplots(figsize=(15,10))
fig19, ax19 = plt.subplots(figsize=(15,10))
fig20, ax20 = plt.subplots(figsize=(15,10))
fig21, ax21 = plt.subplots(figsize=(15,10))

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=0.6, vmax=0.75)

for i, f in enumerate(files, 1):


    env = gym.make('Bixler-v0')

    if ('DQN' in f):
        model = DQN.load(f'{path}{f}')
    elif ('ACKTR' in f):
        env = DummyVecEnv([lambda: env])
        model = ACKTR.load(f'{path}{f}') 
    elif ('A2C' in f):
        env = DummyVecEnv([lambda: env])
        model = A2C.load(f'{path}{f}')
    elif('PPO' in f):
        env = DummyVecEnv([lambda: env])
        model = PPO1.load(f'{path}{f}')
    elif ('TRPO' in f):
        env = DummyVecEnv([lambda: env])
        model = TRPO.load(f'{path}{f}')


    for i in range(100):

        if os.path.exists("../data/output.pkl"):
            os.remove("../data/output.pkl")

        obs = env.reset()

        while True:
            action, _states = model.predict(obs, deterministic = False)
            obs, rewards, done, info = env.step(action)
            if done == True:
                break
            env.render()
        env.close()
    
        reward = rewards

        df = pd.read_pickle('../data/output.pkl')

        final = df.tail(1)

        if reward != -1:

            final.plot.scatter(x = 'pitch' , y = 'u', ax= ax1, color = cmap(norm(reward)))

            final.plot.scatter(x = 'pitch' , y = 'w', ax= ax2, color = cmap(norm(reward)))

            final.plot.scatter(x = 'pitch' , y = 'x', ax= ax3, color = cmap(norm(reward)))

            final.plot.scatter(x = 'pitch' , y = 'z', ax= ax4, color = cmap(norm(reward)))

            final.plot.scatter(x= 'pitch', y = 'airspeed', ax= ax5, color = cmap(norm(reward)))

            final.plot.scatter(x= 'pitch', y = 'q', ax= ax6, color = cmap(norm(reward)))

            final.plot.scatter(x = 'u' , y = 'w', ax= ax7, color = cmap(norm(reward)))

            final.plot.scatter(x = 'u' , y = 'x', ax= ax8, color = cmap(norm(reward)))

            final.plot.scatter(x = 'u' , y = 'z', ax= ax9, color = cmap(norm(reward)))

            final.plot.scatter(x = 'u' , y = 'airspeed', ax= ax10, color = cmap(norm(reward)))

            final.plot.scatter(x = 'u' , y = 'q', ax= ax11, color = cmap(norm(reward)))

            final.plot.scatter(x = 'w' , y = 'x', ax= ax12, color = cmap(norm(reward)))

            final.plot.scatter(x = 'w' , y = 'z', ax= ax13, color = cmap(norm(reward)))

            final.plot.scatter(x = 'w' , y = 'airspeed', ax= ax14, color = cmap(norm(reward)))

            final.plot.scatter(x = 'w' , y = 'q', ax= ax15, color = cmap(norm(reward)))

            final.plot.scatter(x = 'x' , y = 'z', ax= ax16, color = cmap(norm(reward)))

            final.plot.scatter(x = 'x' , y = 'airspeed', ax= ax17, color = cmap(norm(reward)))

            final.plot.scatter(x = 'x' , y = 'q', ax= ax18, color = cmap(norm(reward)))

            final.plot.scatter(x = 'z' , y = 'airspeed', ax= ax19, color = cmap(norm(reward)))

            final.plot.scatter(x = 'z' , y = 'q', ax= ax20, color = cmap(norm(reward)))

            final.plot.scatter(x = 'airspeed' , y = 'q', ax= ax21, color = cmap(norm(reward)))






sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # only needed for matplotlib < 3.1

fig1.colorbar(sm)
fig2.colorbar(sm)   
fig3.colorbar(sm)   
fig4.colorbar(sm)   
fig5.colorbar(sm)   
fig6.colorbar(sm)   
fig7.colorbar(sm)       
fig8.colorbar(sm)   
fig9.colorbar(sm)
fig10.colorbar(sm)  
fig11.colorbar(sm)
fig12.colorbar(sm)   
fig13.colorbar(sm)   
fig14.colorbar(sm)   
fig15.colorbar(sm)   
fig16.colorbar(sm)   
fig17.colorbar(sm)       
fig18.colorbar(sm)   
fig19.colorbar(sm)
fig20.colorbar(sm)
fig21.colorbar(sm)       

plt.show()
