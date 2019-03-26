import gym
import numpy as np
from gym import spaces

class StackObservation(gym.Wrapper):
    def __init__(self, env, stack_size):
        super(StackObservation, self).__init__(env)
        self.memory = [np.zeros(self.env.observation_space.shape) for i in range(stack_size)]
        self.cur_frame = 0
        self.stack_size = stack_size
        print(self.observation_space.low, self.observation_space.high)
        self.observation_space = spaces.Box(low=self.observation_space.low.min(), high=self.observation_space.high.max(),
                dtype=self.observation_space.dtype, shape=self.observation_space.shape + (stack_size,))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.memory[self.cur_frame] = obs
        self.cur_frame = (self.cur_frame + 1) % self.stack_size
        stacked_obs = [self.memory[(self.cur_frame + i) % self.stack_size] for i in range(self.stack_size)]
        stacked_obs = np.moveaxis(np.array(stacked_obs, copy=False), 0, 2)
        return stacked_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.memory[self.cur_frame] = obs
        self.cur_frame = (self.cur_frame + 1) % self.stack_size
        stacked_obs = [self.memory[(self.cur_frame + i) % self.stack_size] for i in range(self.stack_size)]
        stacked_obs = np.moveaxis(np.array(stacked_obs, copy=False), 0, 2)
        return stacked_obs
