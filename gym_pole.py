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
agent = DQNAgent(4, 2, alpha=0.00025, eps_decay_steps=20000, min_history_size=1000, priority_replay=True)
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
        print('i: {} Avg rew: {} Eps: {}'.format(i, np.mean(np.array(l_rewards)), agent.eps))
            



