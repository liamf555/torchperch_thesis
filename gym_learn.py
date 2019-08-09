import gym
#from gym_bixler.envs.bixler_env import BixlerEnv
import gym_bixler


from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN

#env = DummyVecEnv([lambda: BixlerEnv])

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[512, 512, 512, 512],
                                           layer_norm=False,
                                           feature_extraction="mlp")

env = gym.make('Bixler-v0')
model = DQN(CustomPolicy, env, verbose = 0)
model.learn(total_timesteps = 1000)
model.save("deepq_bixler")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        break
    env.render()
env.close()
  
