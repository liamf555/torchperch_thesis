
# from stable_baselines.bench import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
# from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
# from stable_baselines.common.evaluation import evaluate_policy

import os
import warnings

import typing
from typing import Union, List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
# import wandb
from shutil import copyfile

import gym

def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False, path=None):

    # rewards = []
    # wind_speeds = []

    # for env in envs:
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.
    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    # """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths, final_obs = [], [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        # print(env.bixler.airspeed)
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(
                obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render(render)
        final_obs.append(env.env_method("get_final_obs")[0])
        if render is not None:
            env.env_method('save_plots', path=path, reward=episode_reward[0])
        env.close()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    # print(episode_rewards)
    # print(episode_lengths)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_final_obs = [np.mean(i) for i in zip(*final_obs)]
    std_final_obs = [np.std(i) for i in zip(*final_obs)]
    # print(std_final_obs)

    # if reward_threshold is not None:
    #     assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
    #                                     '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    # if return_episode_rewards:
    #     return episode_rewards, episode_lengths
    # return mean_reward, std_reward

    # rewards.append(mean_reward)
    # wind_speeds.append(env.bixler.wind[0])

    rewards = mean_reward
    # wind_speeds = env.bixler.wind[0]

    return rewards, mean_final_obs, std_final_obs
      
class EvalCallback(EventCallback):

    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = [-np.inf] * 3
        self.last_mean_reward = [-np.inf] * 3
        self.deterministic = deterministic
        self.render = render

        # # Convert to VecEnv for consistency
        # if not isinstance(eval_env, VecEnv):
        #     eval_env = DummyVecEnv([lambda: eval_env])

        # assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, wind_speeds = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)


            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            # mean_reward = episode_rewards

            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                wandb.log({"eval_mean_reward": mean_reward})

            if all([mean > best for mean, best in zip(mean_reward, self.best_mean_reward)]):
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

