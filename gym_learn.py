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

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)
	else:
		msg = "Could not find algorithm: {}".format(algorithm_name)
		raise argparse.ArgumentTypeError(msg)


with open("sim_params.json") as json_file:
            params = json.load(json_file)

env = gym.envs.make("Bixler-v0", parameters=params)

ModelType = check_algorithm(params.get("algorithm"))
model = ModelType('MlpPolicy', env, verbose = 1, tensorboard_log=params.get("log_file"))

model.learn(total_timesteps = 1000)

model.save(params.get("model_file"))
