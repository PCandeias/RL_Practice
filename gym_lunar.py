from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
import numpy as np
import gym
from collections import deque
import time

n_episodes = 10000
render = False

env = gym.make('LunarLander-v2')
agent = DQNAgent(8, 4, alpha=0.00025, memory_size=10000, min_history_size=1000, gamma=0.99)
# agent = A2CAgent(8, 4, c_alpha=0.0025, a_alpha=0.0005, memory_size=10000, min_history_size=1000, gamma=0.99)
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
    if ((i+1) % 100) == 0:
        print('i: {} Avg rew: {} N.Replays: {} STEPS: {}'.format(i, np.mean(np.array(l_rewards)), 
            agent.replay_count, agent.cur_step))
            



