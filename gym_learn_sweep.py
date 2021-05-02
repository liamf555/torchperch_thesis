# Filter tensorflow version warnings
import os
import argparse
import json
import wandb
import time

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
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

hyperparameter_defaults = dict(
    env ="Bixler-v0",
    controller = "sweep_elevator_cont_rate",
    scenario = "perching_long_airspeed_2",
    algorithm = "PPO2",
    latency = True,
    noise = 0.0,
    variable_start = True,
    seed  = False,
    timesteps = 5000000,
    start_config = [-40, -5],
    wind_params = [-8, 0],
    wind_mode = "uniform",
    turbulence = "moderate",
    gamma=0.99,
    ent_coef = 0.01,
    net_arch = "medium",
    learning_rate=0.00025,
    cliprange=0.2,
    obs_noise=True
    )

os.environ["WANDB_API_KEY"] = "ea17412f95c94dfcc41410f554ef62a1aff388ab"

wandb.init(sync_tensorboard=True, config=hyperparameter_defaults, dir="/work/tu18537/")
params=wandb.config

def check_algorithm(algorithm_name):
	if hasattr(stable_baselines,algorithm_name):
		return getattr(stable_baselines,algorithm_name)

env = make_vec_env(lambda: gym.make(params.get("env"), parameters=params), n_envs=8, seed=0)
env = VecNormalize(env, norm_reward=False, obs_noise=params.get("obs_noise"))

ModelType = check_algorithm(params.get("algorithm"))

n_steps = params.get("n_steps")
batch_size = params.get("batch_size")

net_arch = params.get("net_arch")

net_arch = {
    "small": [dict(pi=[64, 64], vf=[64, 64])],
    "medium": [dict(pi=[256, 256], vf=[256, 256])],
}[net_arch]

policy_kwargs = dict(
    net_arch=net_arch,
    act_fun=tf.tanh,
)

if n_steps < batch_size:
  nminibatches = 1
else:
  nminibatches = int(n_steps / batch_size)

unixepoch = int(time.time()) 
log_dir = "/work/tu18537/sweep/" + str(unixepoch) + "/"
# log_dir = "../output/sweep/" + str(unixepoch) + "/"
log_dir = Path(log_dir)
log_dir.mkdir(parents=True, exist_ok=True)


model = ModelType('MlpPolicy', env, verbose=0, tensorboard_log=log_dir,
gamma = params.get("gamma"),
ent_coef = params.get("ent_coef"),
learning_rate = params.get("learning_rate"),
noptepochs = params.get("noptepochs"),
cliprange = params.get("cliprange"),
lam = params.get("lam"),
policy_kwargs=policy_kwargs,
nminibatches=nminibatches,
n_steps=n_steps)

# wandb.config.update(params)
# wandb.config.update({"policy": model.policy.__name__})

# for key, value in vars(model).items():
# 	if type(value) == float or type(value) == str or type(value) == int:
# 		wandb.config.update({key: value})
 
model.learn(total_timesteps = wandb.config.timesteps)

model_path = log_dir / 'final_model'

model.save(str(model_path))
vec_file = log_dir / "vec_normalize.pkl"
env.save(str(vec_file))
del model

wind_speeds = [-8.0, -6.0, -4.0, -2.0, 0.0]
mean_rewards = []

for wind in wind_speeds:
    eval_params = dict(params)
    eval_params["wind_params"] = [wind, 0.0, 0.0]
    eval_params["wind_mode"] = "steady"
    eval_params["turbulence"] = "moderate"

    env0 = DummyVecEnv([lambda: gym.make(eval_params.get("env"), parameters=eval_params)])
    eval_env = VecNormalize.load((log_dir / "vec_normalize.pkl"), env0)
    eval_env.training = False

    final_model_path = log_dir / "final_model.zip"
    final_model = ModelType.load(final_model_path)

    final_model.set_env(eval_env)

    final_rewards = evaluate_policy(final_model, eval_env, n_eval_episodes=50, return_episode_rewards=True, render='save')
    mean_rewards.append(final_rewards)

eval_mean_reward = sum(mean_rewards) / len(mean_rewards)
wandb.log({'mean_reward': eval_mean_reward})