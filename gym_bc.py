import gym
import gym_bixler

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.bench import Monitor

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

######DQN########

env = gym.make('Bixler-v0')
model = DQN('MlpPolicy', env, verbose = 0, tensorboard_log="/output/log")


########PPO single core#########

# env = DummyVecEnv([lambda: gym.make('Bixler-v0')])
# # Logs will be saved in log_dir/monitor.csv
# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)

#############PPO multi core###################
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('Bixler-v0') for i in range(n_cpu)])
# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="../log")

start = timeit.default_timer()
model.learn(total_timesteps = 1000000)


model.save("/output/DQN_fixed")
