# imports
# Filter tensorflow version warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)

import json
from pathlib import Path
import gym
import gym_bixler
import argparse
import stable_baselines

from wind.wind_sim import make_eval_wind
from callbacks.callbacks import evaluate_policy


def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)
	else:
		msg = "Could not find algorithm: {}".format(algorithm_name)
		raise argparse.ArgumentTypeError(msg)

def make_eval_envs(params, wind_args=None):

    if wind_args is not None:
        wind_params = wind_args
    else:
        wind_params = params["wind_params"]

    wind_mode = params["wind_mode"]
    wind_north = make_eval_wind(wind_mode, wind_params)
    eval_envs = []

    for wind in wind_north:

        params["wind_params"] = [wind, 0, 0]

        print(params)

        eval_env = gym.make(params.get("env"), parameters = params)
        eval_envs.append(eval_env)

    return eval_envs

parser = argparse.ArgumentParser(description='Load trained model and plot results')
parser.add_argument('--dir_path', type=Path)
parser.add_argument('--render_mode', type = str)
parser.add_argument('--latency', type = float)
parser.add_argument('--noise', type = float)
parser.add_argument('--wind_mode', type = str)
parser.add_argument('--wind_params', type = str, nargs='*')
parser.add_argument('--variable_start', type =str)
args = parser.parse_args()

dir_path = args.dir_path
json_path = args.dir_path / "sim_params.json"

# load json params
with open(json_path) as json_file:
            params = json.load(json_file)

for key, value in vars(args).items():
    if value is not None: 
        if key is not "dir_path":
            if key is not "wind_params":
                params[key] = value
            else:        
                eval_envs = make_eval_envs(params, wind_args=value)
    if key is "wind_params" and value is None:
        eval_envs = make_eval_envs(params)


algorithm = params.get("algorithm")
ModelType = check_algorithm(algorithm)

final_model_path = algorithm + "_final_model.zip"
final_model_path = dir_path / final_model_path

best_model_path = dir_path / 'best_model.zip'

final_model = ModelType.load(final_model_path)
best_model = ModelType.load(best_model_path)

final_rewards, final_wind_speeds = evaluate_policy(final_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True, render='plot')

best_rewards, _ = evaluate_policy(best_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True)


for i, wind in enumerate(final_wind_speeds):
    print(
        f"""Wind(N)(m/s): {wind} -> Final model reward: {final_rewards[i]:.3f}, Best model reward: {best_rewards[i]:.3f}""")