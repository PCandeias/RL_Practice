import gym
import numpy as np
from gym import spaces

class Flatten(gym.Wrapper):
    def __init__(self, env):
        super(Flatten, self).__init__(env)
        self.observation_space = spaces.Box(
                low=self.observation_space.low.min(),
                high=self.observation_space.high.max(), 
                dtype=self.observation_space.dtype,
                shape=(np.prod(env.observation_space.shape),))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.flatten(), reward, done, info

    def reset(self):
        obs = self.env.reset()

        return obs.flatten()
