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

from gym_bixler.envs.bixler_env import BixlerEnv

import argparse

import stable_baselines

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

from stable_baselines import DQN
from stable_baselines import PPO1
from stable_baselines import ACKTR
from stable_baselines import A2C
from stable_baselines import TRPO

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)
	else:
		msg = "Could not find algorithm: {}".format(algorithm_name)
		raise argparse.ArgumentTypeError(msg)

parser = argparse.ArgumentParser(description='Load trained model and plot results')
parser.add_argument('trained_model_file', type=argparse.FileType('r'))
parser.add_argument('--output', type=str, default = '../data/output' )
parser.add_argument('--algorithm', '-a', type=check_algorithm, required=True)
parser.add_argument('--mode', type = str, default = 'plot')
parser.add_argument('--latency', type = float, default = 0.0)
parser.add_argument('--noise', type = float, default = 0.0)
parser.add_argument('--no_var_start', action = 'store_false', dest = 'var_start', default = True)
args = parser.parse_args()

kwargs = {'latency': args.latency, 
          'noise': args.noise,
          'var_start': args.var_start 
          }

env = gym.make('Bixler-v0', **kwargs)

ModelType = args.algorithm
model = ModelType.load(args.trained_model_file.name)
                                                                                   
obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done == True:
        break
    env.render(args.mode)
env.close()