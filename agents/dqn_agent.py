import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
from utility.random_buffer import ReplayBuffer, PriorityReplayBuffer
import utility.utility as utility
import time

class DQNAgent(object):
    def __init__(self, 
                 observation_shape, 
                 action_size, 
                 gamma=1.0, 
                 eps_min=0.1, 
                 eps_eval=0.001,
                 eps_decay_steps=1000000,
                 alpha=5e-4,
                 memory_size=50000,
                 train_frequency=1,
                 batch_size=32, 
                 freeze_target_frequency=500,
                 min_history_size=1000,
                 double_q=True,
                 fixed_q=True,
                 priority_replay=True,
                 verbose=False, 
                 load_filename=None):
        """
        """
        if priority_replay:
            self.memory = PriorityReplayBuffer(max_len=memory_size)
        else:
            self.memory = ReplayBuffer(max_len=memory_size)
        self.min_history_size = min_history_size
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.gamma = gamma
        self.eps = 1.0 
        self.eps_decay = (1.0 - eps_min) / eps_decay_steps
        self.eps_min = eps_min
        self.eps_eval = eps_eval
        self.batch_size = batch_size
        self.verbose = verbose
        self.freeze_target_frequency = freeze_target_frequency
        self.replay_count = 0
        self.eval_mode = False
        self.double_q = double_q
        self.fixed_q = fixed_q
        self.priority_replay = priority_replay
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        priority_beta_decay_steps = 100000
        self.priority_beta_decay = (1.0 - self.priority_beta) / priority_beta_decay_steps
        self._build_model(alpha)
        self.train_frequency = train_frequency
        self.cur_step = 0
        if fixed_q:
            self.target_model = utility.copy_model(self.model) # avoid using target Q-network for first iterations
        else:
            self.target_model = self.model

    def _build_model(self, alpha=0.01):
        self.model = Sequential()
        self.model.add(Dense(units=32, activation='tanh', input_dim=self.observation_shape))
        self.model.add(Dense(units=32, activation='tanh'))
        self.model.add(Dense(units=self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))

    def _load_model(self, load_filename):
        self.model = load_model(utility.models_directory + load_filename + "_dqn.h5")
        self.target_model = utility.copy_model(self.model)

    def _save_model(self, save_filename):
        self.model.save(utility.models_directory + save_filename + "_dqn.h5")

    # get the predictions for a given state
    def _get_predictions(self, observation):
        return self.model.predict(np.array(observation)[np.newaxis,:])

    # select an action according to an eps-greedy policy
    def _select_action(self, observation, eps=None):
        if eps is None:
            eps = self.eps
        return np.random.randint(0, self.action_size) if eps >= np.random.rand() else np.argmax(self._get_predictions(observation))

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
        states = np.array(states, copy=False)
        next_states = np.array(next_states, copy=False)
        y_batch = self.model.predict(states)
        target_pred_after = self.target_model.predict(next_states)
        if self.double_q:
            model_pred_after = self.model.predict(next_states)
            q_values = np.array(rewards, copy=False) + self.gamma * np.invert(np.array(done, copy=False)) * target_pred_after[np.arange(self.batch_size),np.argmax(model_pred_after, axis=1)]
        else:
            q_values = np.array(rewards, copy=False) + self.gamma * np.invert(np.array(done, copy=False)) * np.max(target_pred_after, axis=1)

        # Update priority values
        if self.priority_replay:
            errors = (np.abs(y_batch[np.arange(self.batch_size),actions] - q_values) + 1e-8) ** self.priority_alpha
            for i in range(self.batch_size):
                self.memory.update(idxs[i], errors[i])

        y_batch[np.arange(self.batch_size),actions] = q_values
        # Train the model
        self.model.fit(states, np.array(y_batch), batch_size=self.batch_size, sample_weight=weights, verbose=self.verbose)

        if self.fixed_q and self.replay_count % self.freeze_target_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())

        self.eps = max(self.eps - self.eps_decay, self.eps_min) # update eps
        self.priority_beta = min(self.priority_beta + self.priority_beta_decay, 1.0)

    def _store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def begin_episode(self, observation):
        """
        """
        self.last_observation = observation
        self.last_action = self._select_action(observation, self.eps)
        self.cur_step += 1
        return self.last_action

    def step(self, reward, observation):
        if not self.eval_mode:
            self._store_transition(self.last_observation, self.last_action, reward, observation, False)
            self._train_step()

        self.last_observation = observation
        self.last_action = self._select_action(observation, self.eps if not self.eval_mode else self.eps_eval)
        self.cur_step += 1
        return self.last_action

    def end_episode(self, reward):
        if not self.eval_mode:
            self._store_transition(self.last_observation, self.last_action, reward, self.last_observation, True)
            self._train_step()

class CNNDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(CNNDQNAgent, self).__init__(*args, **kwargs)

    def _build_model(self, alpha=0.01):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=8, strides=4, input_shape=self.observation_shape, activation='relu'))
        self.model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=256, activation='tanh'))
        self.model.add(Dense(units=self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))
