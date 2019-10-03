

import gym
import gym_bixler
import sys

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from stable_baselines import PPO2

import argparse

parser = argparse.ArgumentParser(description='RL for Bixler UAV')
parser.add_argument('--model_file', type=str, default='../output/PPO2')
parser.add_argument('--logfile', type = str, default='../output/logs')
args = parser.parse_args()



# multiprocess environment


def main(args):

    env = gym.make('Bixler-v0')

    n_cpu = 4
    env = SubprocVecEnv([lambda: env for i in range(n_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=args.logfile)


    model.learn(total_timesteps = 2000)

    model.save(args.model_file)

    exit()

if __name__ ==  '__main__':
    main(args)

