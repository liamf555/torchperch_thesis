import gym
import gym_bixler

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from stable_baselines import DQN
from stable_baselines import PPO1
from stable_baselines import ACKTR
from stable_baselines import A2C
from stable_baselines import TRPO


import argparse

parser = argparse.ArgumentParser(description='RL for Bixler UAV')
parser.add_argument('--model_file', type=str, default='../output/DQN_fixed')
parser.add_argument('--algo', type=str, default ='DQN')
parser.add_argument('--logfile', type = str, default='../output/logs')
args = parser.parse_args()


if (args.algo == 'DQN'):
    env = gym.make('Bixler-v0')
    model = DQN('MlpPolicy', env, verbose = 0, tensorboard_log=args.logfile)
elif (args.algo == 'ACKTR'):
    env = gym.make('Bixler-v0')
    env = DummyVecEnv([lambda: env])
    model = ACKTR(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)
elif (args.algo == 'A2C'):
    env = gym.make('Bixler-v0')
    env = DummyVecEnv([lambda: env])
    model = A2C(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)
elif(args.algo == 'PPO'):
    env = gym.make('Bixler-v0')
    env = DummyVecEnv([lambda: env])
    model = PPO1(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)
elif (args.algo == 'TRPO'):
    env = gym.make('Bixler-v0')
    env = DummyVecEnv([lambda: env])
    model = TRPO(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)


model.learn(total_timesteps = 1000000)

model.save(args.model_file)
