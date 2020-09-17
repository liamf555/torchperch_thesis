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

# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.bench import Monitor
# from stable_baselines.common.evaluation import evaluate_policy
from callbacks.callbacks import EvalCallback, evaluate_policy
from wind.wind_sim import make_eval_wind

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

log_dir = params.get("log_file")

wandb.config.update(params)

wandb.config.timesteps=10000000

# env = gym.make(params.get("env"), parameters=params)

env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=8, seed=0, monitor_dir=log_dir)
env = VecFrameStack(env, 4)
env = VecNormalize(env, norm_reward=False)

# eval_envs = make_eval_env(params)

# callback = EvalCallback(eval_envs, eval_freq=1250, log_path=log_dir, best_model_save_path=log_dir, n_eval_episodes=3)

ModelType = check_algorithm(params.get("algorithm"))

model = ModelType("MlpPolicy", env, verbose = 1, tensorboard_log=log_dir)
wandb.config.update({"policy": model.policy.__name__})

for key, value in vars(model).items():
	if type(value) == float or type(value) == str or type(value) == int:
		wandb.config.update({key: value})
 
# model.learn(total_timesteps = wandb.config.timesteps , callback = callback)
model.learn(total_timesteps = wandb.config.timesteps)

model.save(params.get("model_file"))
env.save(log_dir + "vec_normalize.pkl")
wandb.save(log_dir + "vec_normalize.pkl")
wandb.save(params.get("model_file") + ".zip")
wandb.save(log_dir +"best_model.zip")
wandb.save(log_dir + "monitor.csv")

del model

env0 = DummyVecEnv([lambda: gym.make(params.get("env"), parameters=params)])
eval_env = VecNormalize.load((log_dir + "vec_normalize.pkl"), env0)
eval_env.training = False

final_model = ModelType.load(params.get("model_file"))
# best_model = ModelType.load(log_dir +"best_model.zip")

final_model.set_env(eval_env)

final_model_eval = evaluate_policy(final_model, eval_env, n_eval_episodes=1, return_episode_rewards=True, render='save', path = (log_dir+'eval/final_model'))
# best_model_eval = evaluate_policy(best_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True, render='save', path = (log_dir+'eval/best_model'))

final_model_eval = round(final_model_eval, 4)

# best_model_eval = [round(value, 4) for i in final_model_eval for value in i]
# wandb.log({'best_model_eval': best_model_eval})
wandb.log({'final_model_eval': final_model_eval})
wandb.save(log_dir+'eval/*')
# # wand.log({"test_image": wandb.})
