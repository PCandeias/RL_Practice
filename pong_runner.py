from agents.dqn_agent import DQNAgent
from agents.dqn_agent import CNNDQNAgent
from collections import deque
import numpy as np

from wrappers.atari_preprocessing import AtariPreprocessing
from wrappers.stackobservation import StackObservation

import matplotlib.pyplot as plt

from gym_runner import GymRunner
import gym
from keras import backend as K

save_file = 'models/pong'

n_episodes = 100000


def create_agent_fn(sess, env):
    agent = CNNDQNAgent(env.observation_space.shape, env.action_space.n, eps_decay_steps=1000000,  memory_size=100000, min_history_size=10000, freeze_target_frequency=500, train_frequency=4, gamma=0.99, alpha=0.00025)
    return agent

def create_env_fn():
    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = StackObservation(env, 4)
    return env


runner = GymRunner(create_agent_fn, create_env_fn, save_filename=save_file, display_frequency=1)

runner.train(n_episodes)



