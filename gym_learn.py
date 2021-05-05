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

# from callbacks.callbacks import evaluate_policy
from wind.wind_sim import make_eval_wind

from pathlib import Path
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines.common.callbacks import EvalCallback
parser = argparse.ArgumentParser(description='Parse param file location')
parser.add_argument("param_file", type =str, default="sim_params.json")
args = parser.parse_args()

os.environ["WANDB_API_KEY"] = "ea17412f95c94dfcc41410f554ef62a1aff388ab"

# wandb.init(project="disco", sync_tensorboard=True)

wandb.init(project="disco", sync_tensorboard=True, dir="/work/tu18537/")

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)

with open(args.param_file) as json_file:
            params = json.load(json_file)

log_dir = Path(params.get("log_file"))

# env = gym.make(params.get("env"), parameters=params)

env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=8, seed=0)
try: 
	env = VecFrameStack(env, int(params.get("framestack")))
except:
	pass

env = VecNormalize(env, norm_reward=False, obs_noise=params.get("obs_noise"))

net_arch = {
    "small": [dict(pi=[64, 64], vf=[64, 64])],
    "medium": [dict(pi=[256, 256], vf=[256, 256])],
}[params.get("net_arch")]

policy_kwargs = dict(
    net_arch=net_arch,
    act_fun=tf.tanh,
)

ModelType = check_algorithm(params.get("algorithm"))

model = ModelType('MlpPolicy', env, verbose=0, tensorboard_log=log_dir, policy_kwargs = policy_kwargs, n_steps=2048, nminibatches=32) #n_steps=2048, nminibatches=32

# model = ModelType('MlpLstmPolicy', env, verbose=0, tensorboard_log=log_dir)
# model = ModelType(MlpPolicy, env, verbose=0, tensorboard_log=log_dir)

wandb.config.update(params)
wandb.config.update({"policy": model.policy.__name__})

for key, value in vars(model).items():
	if type(value) == float or type(value) == str or type(value) == int:
		wandb.config.update({key: value})


eval_params = params

eval_params["turbulence"] = "light"
eval_params["latency"] = True
eval_params["variable_start"]= True
eval_params["noise"]= 0.3


eval_env = DummyVecEnv([lambda: gym.make(params.get("env"), parameters=eval_params)])

try: 
	eval_env = VecFrameStack(eval_env, int(params.get("framestack")))
except:
	pass

eval_env = VecNormalize(eval_env, norm_reward=False, obs_noise=params.get("obs_noise"), training=False)

eval_env.training = False

eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=25000,
                             deterministic=True, render=False, n_eval_episodes=200)
 
model.learn(total_timesteps = wandb.config.timesteps, callback=eval_callback)

model.save(params.get("model_file"))

vec_file = log_dir / "vec_normalize.pkl"
env.save(str(vec_file))
wandb.save(str(vec_file))
wandb.save(params.get("model_file") + ".zip")
wandb.save(args.param_file)
wandb.save(str(log_dir / "best_model.zip"))
# wandb.save(str(log_dir / "monitor.csv"))

del model

env0 = DummyVecEnv([lambda: gym.make(params.get("env"), parameters=eval_params)])
try: 
	env0 = VecFrameStack(env0, int(params.get("framestack")))
except:
	pass
eval_env = VecNormalize.load(vec_file, env0)
eval_env.training = False

final_model = ModelType.load(params.get("model_file"))
best_model = ModelType.load(str(log_dir / "best_model.zip"))

final_model.set_env(eval_env)
best_model.set_env(eval_env)

final_rewards = evaluate_policy(final_model, eval_env, n_eval_episodes=500, return_episode_rewards=False, render=False)
best_model_rewards = evaluate_policy(best_model, eval_env, n_eval_episodes=500, return_episode_rewards=False, render=False)

wandb.log({'final_model_eval': final_rewards[0]})
wandb.log({'best_model_eval': best_model_rewards[0]})