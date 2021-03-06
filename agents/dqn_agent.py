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
from agents.agent import Agent

class DQNAgent(Agent):
    def __init__(self, 
                 observation_shape, 
                 action_size, 
                 gamma=0.99, 
                 eps_min=0.1, 
                 eps_eval=0.001,
                 eps_decay_steps=100000,
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
        super(DQNAgent, self).__init__(observation_shape, action_size, train_frequency=train_frequency,
                memory_size=memory_size, min_history_size=min_history_size, batch_size=batch_size, verbose=verbose)

        self.gamma = gamma
        self.eps = 1.0 # Initialize eps
        self.eps_decay = (1.0 - eps_min) / eps_decay_steps
        self.eps_min = eps_min
        self.eps_eval = eps_eval

        # Initialize the Replay buffer
        self.priority_replay = priority_replay
        if priority_replay:
            self.memory = PriorityReplayBuffer(max_len=memory_size)
            # Priority replay related parameters
            self.priority_alpha = 0.6
            self.priority_beta = 0.4
            priority_beta_decay_steps = eps_decay_steps
            self.priority_beta_decay = (1.0 - self.priority_beta) / priority_beta_decay_steps
        else:
            self.memory = ReplayBuffer(max_len=memory_size)

        self.double_q = double_q
        self.fixed_q = fixed_q
        self.freeze_target_frequency = freeze_target_frequency

        # Build the models
        self.model = self._build_model(alpha)
        if fixed_q:
            self.target_model = utility.copy_model(self.model) # avoid using target Q-network for first iterations
        else:
            self.target_model = self.model

    def _build_model(self, alpha=0.01):
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_shape=self.observation_shape))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.action_size))
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=alpha))
        return model

    def load_model(self, load_filename):
        self.model = load_model(load_filename + ".h5")
        if self.fixed_q:
            self.target_model = utility.copy_model(self.model)
        else:
            self.target_model = self.model

    def save_model(self, save_filename):
        self.model.save(save_filename + ".h5")

    # get the predictions for a given state
    def _get_predictions(self, observation):
        return self.model.predict(np.array(observation)[np.newaxis,:])

    # select an action according to an eps-greedy policy
    def _select_action(self, observation):
        if self.eval_mode:
            eps = self.eps_eval
        else:
            eps = self.eps
        return np.random.randint(0, self.action_size) if eps >= np.random.rand() else np.argmax(self._get_predictions(observation))

    def _train_step(self):
        if len(self.memory) < self.min_history_size or (self.cur_step % self.train_frequency != 0):
            return

        self.replay_count += 1
        self.eps = max(self.eps - self.eps_decay, self.eps_min) # update eps

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

        y_batch = self.model.predict(states)
        target_pred_after = self.target_model.predict(next_states)

        if self.double_q:
            model_pred_after = self.model.predict(next_states)
            q_values = rewards + self.gamma * np.invert(done) * target_pred_after[np.arange(self.batch_size),np.argmax(model_pred_after, axis=1)]
        else:
            q_values = rewards + self.gamma * np.invert(done) * np.max(target_pred_after, axis=1)

        # Update priority values
        if self.priority_replay:
            errors = (np.abs(y_batch[np.arange(self.batch_size),actions] - q_values) + 1e-8) ** self.priority_alpha
            for i in range(self.batch_size):
                self.memory.update(idxs[i], errors[i])
            # Update priority replay parameters
            self.priority_beta = min(self.priority_beta + self.priority_beta_decay, 1.0)

        y_batch[np.arange(self.batch_size),actions] = q_values

        # Train the model
        self.model.fit(states, y_batch, batch_size=self.batch_size, sample_weight=weights, verbose=self.verbose)

        # Update target model
        if self.fixed_q and self.replay_count % self.freeze_target_frequency == 0:
            self.target_model.set_weights(self.model.get_weights().copy())


class CNNDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(CNNDQNAgent, self).__init__(*args, **kwargs)

    def _build_model(self, alpha=0.01):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8,8), strides=4, input_shape=self.observation_shape, activation='relu'))
        model.add(Conv2D(64, kernel_size=(4,4), strides=2, activation='relu'))
        model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.action_size))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))
        return model
