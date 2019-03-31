
class Agent(object):
    def __init__(self, 
            observation_shape, 
            action_size, 
            gamma=0.99,
            train_frequency=1,
            memory_size=50000,
            min_history_size=1000,
            batch_size=32,
            verbose=False):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.cur_step = 0
        self.replay_count = 0

        self.gamma = gamma

        self.memory_size = memory_size
        self.min_history_size = min_history_size
        self.batch_size = batch_size
        self.train_frequency = train_frequency

        self.verbose = verbose
        self.eval_mode = False

    def _select_action(self, observation):
        raise NotImplementedError("Implement _select_action")

    def _train_step(self):
        raise NotImplementedError("Implement _train_step")

    def _store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def begin_episode(self, observation):
        """
        """
        self.last_observation = observation
        self.last_action = self._select_action(observation)
        self.cur_step += 1

        return self.last_action

    def step(self, reward, observation):
        if not self.eval_mode:
            self._store_transition(self.last_observation, self.last_action, reward, observation, False)
            self._train_step()

        self.last_observation = observation
        self.last_action = self._select_action(observation)
        self.cur_step += 1

        return self.last_action

    def end_episode(self, reward):
        if not self.eval_mode:
            self._store_transition(self.last_observation, self.last_action, reward, self.last_observation, True)
            self._train_step()
