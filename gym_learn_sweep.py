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

from callbacks.callbacks import EvalCallback, evaluate_policy
from wind.wind_sim import make_eval_wind

from pathlib import Path

from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecNormalize

hyperparameter_defaults = dict(
    env ="Bixler-v0",
    controller = "sweep_elevator",
    scenario = "perching_long",
    algorithm = "PPO2",
    latency = 0.0,
    noise = 0.0,
    variable_start = False,
    wind_params = [-8, 0],
    wind_mode = "uniform",
    turbulence ="light",
    seed  = False,
    timesteps = 5000000,
    start_config = [-40, -5],
    gamma=0.99,
    n_steps=128,
    learning_rate=0.00025,
    cliprange=0.2,
    n_batch = 32,
    )

os.environ["WANDB_API_KEY"] = "ea17412f95c94dfcc41410f554ef62a1aff388ab"

wandb.init(sync_tensorboard=True, config=hyperparameter_defaults)
params=wandb.config

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)

env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=8, seed=0)
env = VecNormalize(env, norm_reward=False)

ModelType = check_algorithm(params.get("algorithm"))

n_steps = params.get("n_steps")
batch_size = params.get("n_batch")

if n_steps < batch_size:
  nminibatches = 1
else:
  nminibatches = int(n_steps / batch_size)

log_dir = "/work/tu18537/sweep/"

model = ModelType('MlpPolicy', env, verbose=0, tensorboard_log=log_dir,
gamma = params.get("gamma"),
n_steps = params.get("n_steps"),
learning_rate = params.get("learning_rate"),
noptepochs = params.get("noptepochs"),
cliprange = params.get("cliprange"),
lam = params.get("lam"),
nminibatches = nminibatches)

# wandb.config.update(params)
# wandb.config.update({"policy": model.policy.__name__})

# for key, value in vars(model).items():
# 	if type(value) == float or type(value) == str or type(value) == int:
# 		wandb.config.update({key: value})
 
model.learn(total_timesteps = wandb.config.timesteps)