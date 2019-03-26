import gym

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_steps):
        super(TimeLimit, self).__init__(env)
        self._max_steps = max_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_steps:
            done = True
        return obs, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()
