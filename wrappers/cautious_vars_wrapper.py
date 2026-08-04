"""
cautious_vars_wrapper.py

Turns a CarRacing_RandomSpawn env's raw image observation into the 11-dim
"cautious variables" vector from utils/cautious_variables.py, so a policy can
be trained directly on that vector instead of images -- no CNN, no frame
stacking/skipping. Every physics tick is a real decision (dt_ticks=1).
"""

import gymnasium as gym
import numpy as np

from utils.cautious_variables import CautiousVars


class CautiousVarsWrapper(gym.Wrapper):
    def __init__(self, env, **cautious_kwargs):
        super().__init__(env)
        self.cautious_sensors = CautiousVars(**cautious_kwargs)
        self.n_cautious = 11
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_cautious,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.cautious_sensors.reset_track_reading(self.env)
        obs = self.cautious_sensors.get_cautious_var(self.env, dt_ticks=0)
        return obs, {}

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self.cautious_sensors.get_cautious_var(self.env, dt_ticks=1)
        return obs, reward, terminated, truncated, info
