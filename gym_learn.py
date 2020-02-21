# Filter tensorflow version warnings
import os
import argparse
import json

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

from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from callbacks.callbacks import Callbacks

from stable_baselines import DQN


parser = argparse.ArgumentParser(description='Parse param file location')
parser.add_argument("--param_file", type =str, default="sim_params.json")
args = parser.parse_args()

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)

with open(args.param_file) as json_file:
            params = json.load(json_file)

log_dir = params.get("log_file")

save_cal = Callbacks(log_dir)

env = gym.envs.make("Bixler-v0", parameters=params)
env = Monitor(env, log_dir, allow_early_resets=True)

ModelType = check_algorithm(params.get("algorithm"))

model = ModelType(MlpPolicy, env, verbose = 1)

model.learn(total_timesteps = 1000000, callback = save_cal.auto_save_callback)

model.save(params.get("model_file"))
