import gym
import gym_bixler

import argparse

from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy

parser = argparse.ArgumentParser(description='Load trained model and plot results')
parser.add_argument('--input', type=str, default = '../models/deepq_bixler' )
parser.add_argument('--output', type=str, default = '../data/output' )

args = parser.parse_args()


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[512, 512, 512, 512],
                                           layer_norm=False,
                                           feature_extraction="mlp")

                                           
env = gym.make('Bixler-v0')                                          
model = DQN.load(args.input, policy = CustomPolicy )

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        break
    env.render()
env.close(args.output)