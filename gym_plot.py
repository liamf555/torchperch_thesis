import gym
import gym_bixler

from gym_bixler.envs.bixler_env import BixlerEnv

import argparse

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import DQN
from stable_baselines import PPO1
from stable_baselines import ACKTR
from stable_baselines import A2C
from stable_baselines import TRPO

parser = argparse.ArgumentParser(description='Load trained model and plot results')
parser.add_argument('--input', type=str, default = '../models/deepq_bixler' )
parser.add_argument('--output', type=str, default = '../data/output' )
parser.add_argument('--algo', type = str, default = 'DQN')
parser.add_argument('--mode', type = str, default = 'plot')
parser.add_argument('--latency', type = float, default = 0.0)
parser.add_argument('--noise', type = float, default = 0.0)
args = parser.parse_args()

env = gym.make('Bixler-v0')

# latencies = [0.0, 0.023]
# noises = [0.0, 0.3, 0.5, 1.0]


# env = DummyVecEnv([lambda: BixlerEnv(latency = args.latency, noise = args.noise)])


if (args.algo == 'DQN'):
    model = DQN.load(args.input)
elif (args.algo == 'ACKTR'):
    env = DummyVecEnv([lambda: env])
    model = ACKTR.load(args.input) 
elif (args.algo == 'A2C'):
    env = DummyVecEnv([lambda: env])
    model = A2C.load(args.input)
elif(args.algo == 'PPO'):
    env = DummyVecEnv([lambda: env])
    model = PPO1.load(args.input)
elif (args.algo == 'TRPO'):
    env = DummyVecEnv([lambda: env])
    model = TRPO.load(args.input)


                                                                                    
obs = env.reset()


while True:
    
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done == True:
        break
    env.render(args.mode)
env.close()