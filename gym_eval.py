# imports
# Filter tensorflow version warnings
from matplotlib.pyplot import axis
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
from callbacks.callbacks import evaluate_policy
from wind.wind_sim import make_eval_wind
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import stable_baselines3
import argparse
import gym_bixler
import gym
from pathlib import Path
import json
import logging
import warnings
import os
import numpy as np

def check_algorithm(algorithm_name):
    if hasattr(stable_baselines3, algorithm_name):
        return getattr(stable_baselines3, algorithm_name)
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

        eval_env = gym.make(params.get("env"), parameters=params)
        eval_envs.append(eval_env)

    return eval_envs


parser = argparse.ArgumentParser(
    description='Load trained model and plot results')
parser.add_argument('--dir_path', type=Path)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--latency', action='store_true')
parser.add_argument('--noise', type=float)
parser.add_argument('--wind_mode', type=str)
parser.add_argument('--wind_params', type=str, nargs='*')
parser.add_argument('--turbulence', type=str, default='none')
parser.add_argument('--variable_start', action='store_true')
parser.add_argument('--obs_noise', action='store_true')
args = parser.parse_args()

dir_path = args.dir_path
json_path = args.dir_path / "sim_params.json"

# load json params
with open(json_path) as json_file:
    params = json.load(json_file)

algorithm = params.get("algorithm")
ModelType = check_algorithm(algorithm)

final_model_path = algorithm + "_final_model.zip"
final_model_path = dir_path / final_model_path
best_model_path = dir_path / 'best_model.zip'


wind_speeds = [-8.0, -6.0, -4.0, -2.0, 0.0]
final_mean_rewards = []
best_mean_rewards = []
mean_final_states_array = []
std_final_states_array = []

for wind in wind_speeds:

    # wind = -4.0

    params["wind_params"] = [wind, 0.0, 0.0]
    params["wind_mode"] = "steady"
    params["turbulence"] = args.turbulence
    params["latency"] = args.latency
    params["variable_start"] = args.variable_start
    params["noise"] = args.noise
    params["obs_noise"] = args.obs_noise

    print(params)

    # print(wind)

    env0 = DummyVecEnv(
        [lambda: gym.make(params.get("env"), parameters=params)])

    try:
        env0 = VecFrameStack(env0, int(params.get("framestack")))
    except:
        pass

    eval_env = VecNormalize.load((dir_path / "vec_normalize.pkl"), env0)
    eval_env.training = False
    eval_env.obs_noise = params.get("obs_noise")

    final_model = ModelType.load(final_model_path)
    best_model = ModelType.load(best_model_path)

    final_model.set_env(eval_env)

    # final_model_eval = evaluate_policy(final_model, eval_env, n_eval_episodes=1, return_episode_rewards=True, render='save', path = (log_dir+'eval/final_model'))

    eval_dir = dir_path / ('eval_' + str(wind))

    eval_dir.mkdir(parents=True, exist_ok=True)

    final_model_data_path = eval_dir / 'final_model'
    best_model_data_path = eval_dir / 'best_model'

    final_rewards, mean_final_states, std_final_states = evaluate_policy(
        final_model, eval_env, n_eval_episodes=1, return_episode_rewards=True, render=args.render_mode, path=str(final_model_data_path))
    # best_rewards, _, _ = evaluate_policy(best_model, eval_env, n_eval_episodes=1,
    #  return_episode_rewards = True, render = args.render_mode, path = str(best_model_data_path))

    final_mean_rewards.append(final_rewards)
    # best_mean_rewards.append(best_rewards)
    mean_final_states_array.append(mean_final_states)
    std_final_states_array.append(std_final_states)

    # print(mean_final_states)

    # best_rewards, best_reward_speeds = evaluate_policy(best_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True, render=args.render_mode, path = best_model_data_path)

    # final_rewards, final_wind_speeds = evaluate_policy(final_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True)

    # best_rewards, best_reward_speeds = evaluate_policy(best_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True )

    # print(final_rewards)

df = pd.DataFrame(mean_final_states_array, columns=[
    "x", "z", "theta", "u", "w"])

df_2 = pd.DataFrame(std_final_states_array, columns=[
    "x", "z", "theta", "u", "w"])

# df["theta"] = np.rad2deg(df["theta"])

df = pd.concat([df, df_2], axis=1)

headers = []

headers.extend([("test"), str("test_2")])

df.columns = pd.MultiIndex.from_tuples(
    zip(['Mean', 'Mean', 'Mean', 'Mean', 'Mean', 'SD', 'SD', 'SD', 'SD', 'SD'],
        df.columns))


df.to_csv(str(dir_path / "final_states.csv"))


print("Final model")

for reward in final_mean_rewards:
    print(reward)

# for states in mean_final_states_array:
#     print(states)
# plt.show()
