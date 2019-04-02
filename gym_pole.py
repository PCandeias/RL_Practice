from agents.a2c_agent import A2CAgent
from agents.dqn_agent import DQNAgent
import numpy as np
import gym
from collections import deque
import time
from wrappers.timelimit import TimeLimit

n_episodes = 20000
render = False

env = gym.make('CartPole-v1')
env = TimeLimit(env, 200)
agent = DQNAgent((4,), 2, memory_size=10000, eps_decay_steps=10000, min_history_size=100, alpha=.00025)
# agent = A2CAgent(4, 2, memory_size=10000, min_history_size=100, c_alpha=0.0025, a_alpha=0.0005, priority_replay=True)
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
        print('i: {} Avg rew: {} ReplayCount: {}'.format(i, np.mean(np.array(l_rewards)), agent.replay_count))
            



