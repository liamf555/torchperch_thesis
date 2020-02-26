# Filter tensorflow version warnings
import os
import argparse
import json
import wandb
import inspect

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

from stable_baselines import DQN, PPO2

parser = argparse.ArgumentParser(description='Parse param file location')
parser.add_argument("--param_file", type =str, default="sim_params.json")
args = parser.parse_args()

os.environ["WANDB_API_KEY"] = "ea17412f95c94dfcc41410f554ef62a1aff388ab"

wandb.init(project="disco", sync_tensorboard=True)

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)

with open(args.param_file) as json_file:
            params = json.load(json_file)

log_dir = params.get("log_file")

wandb.config.update(params)
wandb.config.timesteps=1000

save_cal = Callbacks(log_dir)

env = gym.envs.make(params.get("env"), parameters=params)
env = Monitor(env, log_dir, allow_early_resets=True)

ModelType = check_algorithm(params.get("algorithm"))
model = ModelType(MlpPolicy, env, verbose = 1, tensorboard_log=log_dir)
wandb.config.update({"policy": model.policy.__name__})

for key, value in vars(model).items():
	if type(value) == float or type(value) == str or type(value) == int:
		wandb.config.update({key: value})


model.learn(total_timesteps = wandb.config.timesteps , callback = save_cal.auto_save_callback)

model.save(params.get("model_file"))
wandb.save(params.get("model_file") + ".zip")
wandb.save(log_dir +"/best_model.zip")
wandb.save(log_dir + "/monitor.csv")

final_model = ModelType.load(params.get("model_file"))
best_model = ModelType.load(log_dir +"/best_model.zip")
final_model_eval = evaluate_policy(final_model, env)
best_model_eval = evaluate_policy(best_model, env)
wandb.log({'best_model_eval': best_model_eval})
wandb.log({'final_model_eval': final_model_eval})

