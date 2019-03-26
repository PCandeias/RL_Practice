from agents.dqn_agent import CNNDQNAgent
import numpy as np
import gym
from collections import deque
import time
from keras import backend as K
from wrappers.atari_preprocessing import AtariPreprocessing
from wrappers.stackobservation import StackObservation

import matplotlib.pyplot as plt


n_episodes = 100000
render = False

env = gym.make('Breakout-v0')
env = AtariPreprocessing(env)
print(env.observation_space.shape)
env = StackObservation(env, 4)
print(env.observation_space.shape)

agent = CNNDQNAgent(env.observation_space.shape, env.action_space.n)
l_rewards = deque(maxlen=100)

for i in range(n_episodes):
    done = False
    obs = env.reset()
    action = agent.begin_episode(obs)
    total_reward = 0
    while not done:
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            agent.end_episode(reward)
        else:
            action = agent.step(reward, obs)
    l_rewards.append(total_reward)
    if ((i+1) % 1) == 0:
        print('i: {} Avg rew: {} Eps: {}'.format(i, np.mean(np.array(l_rewards)), agent.eps))
