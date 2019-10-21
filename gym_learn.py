import gym
#import gym_bixler

from gym_bixler.envs.bixler_env import BixlerEnv

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
parser.add_argument('--latency', type = float, default = 0.0)
parser.add_argument('--noise', type = float, default = 0.0)
args = parser.parse_args()

kwargs = {'latency': args.latency, 
          'noise': args.noise,
          }

# env = gym.make('Bixler-v0', **kwargs)
env = DummyVecEnv([lambda: BixlerEnv(latency = args.latency, noise = args.noise)])


if (args.algo == 'DQN'):
    model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=args.logfile)
elif (args.algo == 'ACKTR'):
    env = DummyVecEnv([lambda: env])
    model = ACKTR(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)
elif (args.algo == 'A2C'):
    env = DummyVecEnv([lambda: env])
    model = A2C(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)
elif(args.algo == 'PPO'):
    env = DummyVecEnv([lambda: env])
    model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log=args.logfile)
elif (args.algo == 'TRPO'):
    env = DummyVecEnv([lambda: env])
    model = TRPO(MlpPolicy, env, verbose=0, tensorboard_log=args.logfile)


model.learn(total_timesteps = 10000)

model.save(args.model_file)
