import gym
from gym import spaces
import numpy as np

class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env):
        super(AtariPreprocessing, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(80,80))

    def __preprocessing(self, observation):
        return np.mean(observation, axis=2).astype('uint8')[35:195][::2,::2]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.__preprocessing(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.__preprocessing(obs)
        return obs
