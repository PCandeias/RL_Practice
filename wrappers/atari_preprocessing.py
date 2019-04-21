import gym
from gym import spaces
import numpy as np


class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env):
        super(AtariPreprocessing, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(80,80))
        self.last_obs = np.zeros((80,80))

    def __preprocessing(self, observation):
        observation = observation[35:195]
        observation = observation[::2,::2,0]
        observation[observation == 144] = 0
        observation[observation == 109] = 0
        observation[observation != 0] = 1

        n_observation = np.max(np.array([observation, self.last_obs]), axis=0)
        self.last_obs = observation

        return n_observation.astype(np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.__preprocessing(obs)
        return obs, reward, done, info

    def reset(self):
        self.last_obs = np.zeros((80,80))
        obs = self.env.reset()
        obs = self.__preprocessing(obs)
        return obs
