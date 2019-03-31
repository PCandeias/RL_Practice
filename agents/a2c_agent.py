import numpy as np
from keras.utils import np_utils
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
from utility.random_buffer import ReplayBuffer, PriorityReplayBuffer
import utility.utility as utility
import time
from agents.agent import Agent

class A2CAgent(Agent):
    def __init__(self, 
                 observation_shape, 
                 action_size, 
                 gamma=0.99, 
                 c_alpha=0.00025,
                 a_alpha=0.00025,
                 memory_size=50000,
                 batch_size=32, 
                 train_frequency=1,
                 min_history_size=1000,
                 priority_replay=True,
                 verbose=False, 
                 load_filename=None):
        """
        """
        super(A2CAgent, self).__init__(observation_shape, action_size, gamma=gamma, memory_size=memory_size, min_history_size=min_history_size, batch_size=batch_size, train_frequency=train_frequency, verbose=verbose)

        # Initialize the Replay buffer
        self.priority_replay = priority_replay
        if priority_replay:
            self.memory = PriorityReplayBuffer(max_len=memory_size)
            # Priority replay related parameters
            self.priority_alpha = 0.6
            self.priority_beta = 0.4
            priority_beta_decay_steps = 100000
            self.priority_beta_decay = (1.0 - self.priority_beta) / priority_beta_decay_steps
        else:
            self.memory = ReplayBuffer(max_len=memory_size)

        # Build the models
        self.critic = self._build_critic_model(c_alpha)
        self.actor = self._build_actor_model(a_alpha)

    def _build_critic_model(self, alpha=0.01):
        model = Sequential()
        model.add(Dense(units=32, activation='tanh', input_dim=self.observation_shape))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))
        return model

    def _build_actor_model(self, alpha=0.01):
        model = Sequential()
        model.add(Dense(units=32, activation='tanh', input_dim=self.observation_shape))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=alpha))
        return model

    def _load_model(self, load_filename):
        self.actor = load_model(utility.models_directory + load_filename + "_ac_a2c.h5")
        self.critic = load_model(utility.models_directory + load_filename + "_cr_a2c.h5")

    def _save_model(self, save_filename):
        self.actor.save(utility.models_directory + save_filename + "_ac_a2c.h5")
        self.critic.save(utility.models_directory + save_filename + "_cr_a2c.h5")

    # get the predictions for a given state
    def _get_predictions(self, observation):
        return self.critic.predict(np.array(observation)[np.newaxis,:])

    # select an action according to an eps-greedy policy
    def _select_action(self, observation):
        ps = self.actor.predict(np.array([observation]))[0]
        return np.random.choice(self.action_size, p=ps)

    def _train_step(self):
        if len(self.memory) < self.min_history_size or (self.cur_step % self.train_frequency != 0):
            return

        self.replay_count += 1

        # Get a batch of state-transitions
        if self.priority_replay:
            mini_batch, idxs, weights = self.memory.sample(self.batch_size)
            weights = weights ** self.priority_beta
        else:
            mini_batch, idxs = self.memory.sample(self.batch_size)
            weights = np.ones(self.batch_size)

        states, actions, rewards, next_states, done = zip(*mini_batch)
        states, actions, rewards, next_states, done = np.array(states, copy=False), np.array(actions, copy=False), \
                np.array(rewards, copy=False), np.array(next_states, copy=False), np.array(done, copy=False)

        pred_before = self.critic.predict(states)
        pred_after = self.critic.predict(next_states)

        c_y_batch = np.invert(done)[:,np.newaxis] * self.gamma * pred_after + rewards[:,np.newaxis]
        a_y_batch = np_utils.to_categorical(actions, self.action_size)

        errors = (c_y_batch - pred_before)[:,0]

        a_weights = weights * errors

        # Update priority values
        if self.priority_replay:
            a_errors = np.abs(errors)
            for i in range(self.batch_size):
                e = (a_errors[i] + 1e-8) ** self.priority_alpha
                self.memory.update(idxs[i], e)
            # Update priority replay parameters
            self.priority_beta = min(self.priority_beta + self.priority_beta_decay, 1.0)


        # Train the model
        self.actor.fit(states, a_y_batch, batch_size=self.batch_size, sample_weight=a_weights, verbose=self.verbose)
        self.critic.fit(states, c_y_batch, batch_size=self.batch_size, sample_weight=weights, verbose=self.verbose)
