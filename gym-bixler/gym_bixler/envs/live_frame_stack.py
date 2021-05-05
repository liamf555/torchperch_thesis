import numpy as np
from gym import spaces

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper

class LiveFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment dureing real world testing

    :param venv: (VecEnv) the vectorized environment to wrap
    :param n_stack: (int) Number of frames to stack
    """

    def __init__(self, venv, n_stack):
        self.venv = venv
        self.n_stack = n_stack
        wrapped_obs_space = venv.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)


    def stack_obs(self, transformed_obs):
        last_ax_size = transformed_obs.shape[-1]
        self.stackedobs = np.roll(self.stackedobs, shift=-last_ax_size, axis=-1)
        self.stackedobs[..., -transformed_obs.shape[-1]:] = transformed_obs
        return self.stackedobs

    def step_wait(self):
        return super().step_wait()

    def reset(self):
        return super().reset()

    def stack_reset(self):
        """
        Reset all environments
        """
        self.stackedobs[...] = 0
        # self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()