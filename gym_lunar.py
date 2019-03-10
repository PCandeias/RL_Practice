from agents.dqn_agent import DQNAgent
import numpy as np
import gym
from collections import deque
import time

n_episodes = 2000
render = False

env = gym.make('LunarLander-v2')
agent = DQNAgent(8, 4, alpha=0.00025, eps_decay_steps=300000)
l_rewards = deque(maxlen=100)

for i in range(n_episodes):
    done = False
    obs = env.reset()
    action = agent.begin_episode(obs)
    total_reward = 0
    while not done:
        if i >= 2000 and render:
            env.render()
            time.sleep(0.1)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            agent.end_episode(reward)
        else:
            action = agent.step(reward, obs)
    l_rewards.append(total_reward)
    if ((i+1) % 100) == 0:
        print('i: {} Avg rew: {} Eps: {}'.format(i, np.mean(np.array(l_rewards)), agent.eps))
            


