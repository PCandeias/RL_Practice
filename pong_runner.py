from agents.dqn_agent import CNNDQNAgent
import numpy as np
import gym
from collections import deque
import time
from keras import backend as K



def preprocess_obs(obs):
    return np.mean(obs, axis=2).astype('uint8')[35:195][::2,::2]

n_episodes = 100
render = False

env = gym.make('Pong-v0')
print(env.observation_space.shape)
agent = CNNDQNAgent((80,80), env.action_space.n, train_frequency=400)
l_rewards = deque(maxlen=100)

for i in range(n_episodes):
    done = False
    obs = env.reset()
    obs = preprocess_obs(obs)
    action = agent.begin_episode(obs)
    total_reward = 0
    while not done:
        if i >= 2000 and render:
            env.render()
            time.sleep(0.1)
        obs, reward, done, info = env.step(action)
        obs = preprocess_obs(obs)
        total_reward += reward
        if done:
            agent.end_episode(reward)
        else:
            action = agent.step(reward, obs)
    l_rewards.append(total_reward)
    if ((i+1) % 1) == 0:
        print('i: {} Avg rew: {} Eps: {}'.format(i, np.mean(np.array(l_rewards)), agent.eps))
