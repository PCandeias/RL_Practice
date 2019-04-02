import numpy as np
from collections import deque
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.models import load_model, Model
from keras.losses import mse
from utility.random_buffer import ReplayBuffer, PriorityReplayBuffer
import utility.utility as utility
import time
from agents.agent import Agent
from keras import backend as K

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
        super(DQNAgent, self).__init__(observation_shape, action_size, gamma=gamma, train_frequency=train_frequency,
                memory_size=memory_size, min_history_size=min_history_size, batch_size=batch_size, verbose=verbose)
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.cur_step = 0
        self.replay_count = 0

        self.verbose = verbose
        self.eval_mode = False
        self.train_frequency = train_frequency
        self.batch_size = batch_size

        self.gamma = gamma
        self.eps = 1.0 # Initialize eps
        self.eps_decay = (1.0 - eps_min) / eps_decay_steps
        self.eps_min = eps_min
        self.eps_eval = eps_eval

        # Initialize the Replay buffer
        self.min_history_size = min_history_size
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

        self.double_q = double_q,
        self.fixed_q = fixed_q
        self.freeze_target_frequency = freeze_target_frequency

        # Build the models
        self.model, self.train_model = self._build_model(alpha)
        if fixed_q:
            self.target_model = utility.copy_model(self.model) # avoid using target Q-network for first iterations
        else:
            self.target_model = self.model

    def _build_model(self, alpha=0.01):
        x = Input(shape=self.observation_shape)
        y_true = Input(shape=(self.action_size,))
        weights = Input(shape=(1,))
        f = Dense(units=200, activation='relu')(x)
        f = Dense(units=200, activation='relu')(f)
        y_pred = Dense(units=self.action_size)(f)

        def weighted_loss(y_true, y_pred, weights):
            return K.sum(mse(y_true, y_pred) * weights[:,0])

        train_model = Model(inputs=[x, y_true, weights], outputs=y_pred)
        model = Model(inputs=x, outputs=y_pred)

        train_model.add_loss(weighted_loss(y_true, y_pred, weights))
        # train_model.compile(loss=None, optimizer=keras.optimizers.Adam(lr=alpha))
        train_model.compile(loss=None, optimizer=keras.optimizers.Adam(lr=alpha), metrics=['mse'])
        # model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))

        return model, train_model
        # return model, model

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
        # self.model.fit(states, y_batch, batch_size=self.batch_size, sample_weight=weights, verbose=self.verbose)
        a = self.train_model.fit([states, y_batch, weights], batch_size=self.batch_size, verbose=self.verbose)

        # Update target model
        if self.fixed_q and self.replay_count % self.freeze_target_frequency == 0:
            self.target_model.set_weights(self.model.get_weights().copy())


class CNNDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(CNNDQNAgent, self).__init__(*args, **kwargs)

    def _build_model(self, alpha=0.01):
        x = Input(shape=self.observation_shape)
        y_true = Input(shape=(self.action_size,))
        weights = Input(shape=(1,))
        f = Conv2D(32, kernel_size=8, strides=4, activation='relu')(x)
        f = Conv2D(64, kernel_size=4, strides=2, activation='relu')(f)
        f = Conv2D(64, kernel_size=3, strides=1, activation='relu')(f)
        f = Flatten()(f)
        f = Dense(units=256, activation='relu')(f)
        y_pred = Dense(units=self.action_size)(f)

        def weighted_loss(y_true, y_pred, weights):
            return K.sum(mse(y_true, y_pred) * weights[:,0])

        train_model = Model(inputs=[x, y_true, weights], outputs=y_pred)
        model = Model(inputs=x, outputs=y_pred)

        train_model.add_loss(weighted_loss(y_true, y_pred, weights))
        train_model.compile(loss=None, optimizer=keras.optimizers.Adam(lr=alpha), metrics=['mse'])

        return model, train_model
