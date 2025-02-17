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
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines import HER, DQN, SAC, DDPG

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

		eval_env = gym.make(params.get("env"), parameters = params)

		eval_envs.append(eval_env)

	return eval_envs

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
wandb.config.timesteps=1000000

env = gym.make(params.get("env"), parameters=params)

# env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=1, seed=0, monitor_dir=log_dir)

# env = VecNormalize(env, norm_reward=False)

# eval_envs = make_eval_env(params)

# callback = EvalCallback(eval_envs, eval_freq=1250, log_path=log_dir, best_model_save_path=log_dir, n_eval_episodes=3)

ModelType = check_algorithm(params.get("algorithm"))

# model = ModelType("MlpPolicy", env, verbose = 1, tensorboard_log=log_dir)

goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

model_class = DQN

# Wrap the model
model = HER('MlpPolicy', env, model_class,verbose=1, tensorboard_log=log_dir)

# wandb.config.update({"policy": model.policy.__name__})

for key, value in vars(model).items():
	if type(value) == float or type(value) == str or type(value) == int:
		wandb.config.update({key: value})

# model.learn(total_timesteps = wandb.config.timesteps , callback = callback)
model.learn(total_timesteps = wandb.config.timesteps, tb_log_name=log_dir + 'HER')

model.save(params.get("model_file"))
wandb.save(params.get("model_file") + ".zip")
wandb.save(log_dir +"best_model.zip")
wandb.save(log_dir + "monitor.csv")

# final_model = ModelType.load(params.get("model_file"))
model = HER.load(params.get("model_file"), env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done == True:
        print("Reward:", episode_reward, "Success?", info.get('is_success', False))
        episode_reward = 0.0
        break
    env.render(mode='plot')
env.close(path = log_dir+'eval/final_model' , reward = 0)




# # final_model_eval = evaluate_policy(final_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True, render='save', path = (log_dir+'eval/final_model'))
# best_model_eval = evaluate_policy(best_model, eval_envs, n_eval_episodes=1, return_episode_rewards=True, render='save', path = (log_dir+'eval/best_model'))
# best_model_eval = [round(value, 4) for i in best_model_eval for value in i]
# wandb.log({'best_model_eval': best_model_eval})
# # wandb.log({'final_model_eval': final_model_eval})
# wandb.save(log_dir+'eval/*')
# # wand.log({"test_image": wandb.})
