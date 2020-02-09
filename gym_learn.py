# Filter tensorflow version warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

import gym
import gym_bixler
import stable_baselines

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from stable_baselines import DQN
from stable_baselines import PPO2

# import argparse

import json

# def check_algorithm(algorithm_name):
# 	if hasattr(stable_baselines,algorithm_name):
# 		return getattr(stable_baselines,algorithm_name)
# 	else:
# 		msg = "Could not find algorithm: {}".format(algorithm_name)
# 		raise argparse.ArgumentTypeError(msg)

# parser = argparse.ArgumentParser(description='RL for Bixler UAV')
# parser.add_argument('--model_file', type=str, default='../output/DQN')
# parser.add_argument('--algorithm', '-a', type=check_algorithm, required=True)
# parser.add_argument('--logfile', type = str, default='../output/logs')
# parser.add_argument('--latency', type = float, default = 0.0)
# parser.add_argument('--noise', type = float, default = 0.0)
# parser.add_argument('--no_var_start', action = 'store_false', dest = 'var_start', default = True)
# parser.add_argument('--wind', type =  )

# args = parser.parse_args()

# kwargs = {'latency': args.latency, 
#           'noise': args.noise,
#           'var_start': args.var_start 
        #   }

with open("sim_params.json") as json_file:
    params = json.load(json_file)

kwargs = {'latency': params.latency, 
          'noise': params.noise,
          'var_start': params.variable_start 
          }

env = gym.make('Bixler-v0', **kwargs)

env.seed(1995)

ModelType = args.algorithm
model = ModelType('MlpPolicy', env, verbose = 1, tensorboard_log=args.logfile)

model.learn(total_timesteps = 1000)

model.save(args.model_file)
