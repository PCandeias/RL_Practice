import numpy as np
from agents.dqn_agent import DQNAgent

agent = DQNAgent(1,9, decay_steps=20000)

for i in range(100000):
    print("Iteration {}".format(i))
    cur_it = 0
    a = agent.begin_episode(np.ones(1) * 0)
    val = np.random.randint(0,9)
    reward = 1 if a == val else 0
    print('EPS: {}'.format(agent.eps)) 
    print("ACTION: {} REWARD: {} J: {}".format(a, reward, val))
    for j in range(1, 10):
        if reward == 0:
            break
        else:
            val = np.random.randint(0,9)
            a = agent.step(reward, np.ones(1) * val)
            reward = 1 if a == val else 0
            print("ACTION: {} REWARD: {} J: {}".format(a, reward, val))
    agent.end_episode(reward)

for i in range(9):
    print('i: {} pred: {} action:{}'.format(i, agent._get_predictions(np.ones(1) * i), agent._select_action(np.ones(1) * i, eps=0)))
