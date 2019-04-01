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

env = gym.make('PongNoFrameskip-v4')
env = AtariPreprocessing(env)
print(env.observation_space.shape)
env = StackObservation(env, 4)
print(env.observation_space.shape)

agent = CNNDQNAgent(env.observation_space.shape, env.action_space.n, eps_decay_steps=100000,  memory_size=100000,
        min_history_size=10000, freeze_target_frequency=500, train_frequency=4, gamma=0.99, alpha=0.0001, double_q=False, fixed_q=False, priority_replay=False)

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
