# Filter tensorflow version warnings
import os
import argparse
import json
import wandb

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

# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.bench import Monitor
# from stable_baselines.common.evaluation import evaluate_policy
from callbacks.callbacks import EvalCallback, evaluate_policy
from wind.wind_sim import make_eval_wind

from pathlib import Path

# from stable_baselines.common.cmd_util import make_vec_env

# from stable_baselines import DQN, PPO2, SAC

from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack

def make_eval_env(params):

	if params.get("wind_mode") == 'normal':
		params["wind_mode"] = 'normal_eval'   
		wind_north = make_eval_wind("normal_eval", params["wind_params"])

	if params.get("wind_mode") == "steady":
		params["wind_mode"] = 'steady_eval'   
		wind_north = make_eval_wind("steady_eval", params["wind_params"])

	eval_envs = []

	for wind in wind_north:
		
		params["wind_params"] = [wind, 0, 0]

		# eval_env = gym.make(params.get("env"), parameters = params)
		eval_env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=8, seed=0)

		eval_envs.append(eval_env)

	return eval_envs

parser = argparse.ArgumentParser(description='Parse param file location')
parser.add_argument("param_file", type =str, default="sim_params.json")
args = parser.parse_args()

os.environ["WANDB_API_KEY"] = "ea17412f95c94dfcc41410f554ef62a1aff388ab"

wandb.init(project="disco", sync_tensorboard=True)

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)

with open(args.param_file) as json_file:
            params = json.load(json_file)

log_dir = Path(params.get("log_file"))

# env = gym.make(params.get("env"), parameters=params)

env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=16, seed=0, monitor_dir=log_dir)
env = VecNormalize(env, norm_reward=False)

ModelType = check_algorithm(params.get("algorithm"))

model = ModelType('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

wandb.config.update(params)
wandb.config.update({"policy": model.policy.__name__})

for key, value in vars(model).items():
	if type(value) == float or type(value) == str or type(value) == int:
		wandb.config.update({key: value})
 
model.learn(total_timesteps = wandb.config.timesteps)

model.save(params.get("model_file"))

vec_file = log_dir / "vec_normalize.pkl"
env.save(str(vec_file))
wandb.save(str(vec_file))
wandb.save(params.get("model_file") + ".zip")
wandb.save(str(log_dir / "best_model.zip"))
wandb.save(str(log_dir / "monitor.csv"))

del model

params["training"] = False


env0 = DummyVecEnv([lambda: gym.make(params.get("env"), parameters=params)])
eval_env = VecNormalize.load((vec_file), env0)
eval_env.training = False

final_model = ModelType.load(params.get("model_file"))

final_model.set_env(eval_env)

final_model_eval = evaluate_policy(final_model, eval_env, n_eval_episodes=1, return_episode_rewards=True, render='save', path = str(log_dir/ 'eval/final_model'))

final_model_eval = round(final_model_eval, 4)

wandb.log({'final_model_eval': final_model_eval})
wandb.save(str(log_dir/'eval/*'))
