import tensorflow as tf
import collections
import numpy as np
import datetime
import pickle
from collections import deque

class GymRunner(object):
    def __init__(self, 
                 create_agent_fn, 
                 create_environment_fn, 
                 log_observation=True,
                 render=False,
                 display_frequency=100,
                 args=None,
                 verbose=False,
                 save_filename=None,
                 log_dir=None):
        self.env = create_environment_fn()
        sess = tf.Session()
        self.start_time = datetime.datetime.now()
        self.display_frequency = display_frequency
        self.verbose = verbose
        self.agent = create_agent_fn(sess, self.env)
        self.log_dir = log_dir
        self.memory = []
        self.ep_memory = {}
        self.ep_number = 0
        self.render = render
        self.log_observation=log_observation
        self.args = args # Just for having the parameters to log
        self.l_rewards = deque(maxlen=100)
        self.b_mean = -1000000
        self.save_filename = save_filename
        sess.run(tf.global_variables_initializer())


    def _run_one_episode(self):
        # Initialize episode
        is_over = False
        step_number = 0
        observation = self.env.reset()

        ep_memory = {'b_observation':[], 'a_observation':[], 'rewards':[], 'is_over':[],  'debug_data':[], 'actions':[]}

        action = self.agent.begin_episode(observation)

        t_reward = 0

        while not is_over:
            observation, reward, is_over, metrics = self._run_one_step(observation, action, ep_memory)
            step_number += 1
            t_reward += reward

            if is_over:
                self.agent.end_episode(reward)
            else:
                action = self.agent.step(reward, observation)
        self.l_rewards.append(t_reward)

        if self.log_dir:
            metrics = self._compute_metrics(ep_memory)
            ep_memory['episode_number'] = self.ep_number
            ep_memory['metrics'] = metrics
            self.memory.append(ep_memory)

        self.ep_number += 1

        return step_number
        
    def _run_one_step(self, l_observation, actions, ep_memory):
        """
        Run one step in the environment

        Returns:

        """
        observation, reward, is_over, debug_data = self.env.step(actions)
        if self.render:
            self.env.render()
        if self.log_dir:
            ep_memory['actions'].append(actions)
            if self.log_observation:
                ep_memory['b_observation'].append(l_observation)
                ep_memory['a_observation'].append(observation)
                ep_memory['debug_data'].append(debug_data)
                ep_memory['is_over'].append(is_over)
            ep_memory['rewards'].append(rewards)
        return observation, reward, is_over, debug_data

    def _compute_metrics(self, ep_memory):
        return {'R':R, 'U':U, 'E':E}

    def _save_log(self):
        if self.log_dir:
            stored = {'start':self.start_time, 'args': self.args, 'episodes':self.memory}
            file_str = '{0}_{1:%H_%M_%S_%d%m%Y}'.format(self.n_agents, self.start_time)
            path = self.log_dir + '/' + file_str
            with open(path, "wb") as fp:
                pickle.dump(stored, fp)
            print('Saved results to {}'.format(path))

    def train(self, n_episodes):
        eval_mode = self.agent.eval_mode
        self.agent.eval_mode = False

        total_steps = 0
        for e in range(n_episodes):
            total_steps += self._run_one_episode()
            mean = np.mean(np.array(self.l_rewards))
            if self.save_filename and self.ep_number >= 500 and mean > self.b_mean:
                print("Model outperformed mean. Prev mean: {0} New mean: {1}".format(self.b_mean, mean))
                self.agent.save_model(self.save_filename)
                self.b_mean = mean

            if (e+1) % self.display_frequency == 0:
                print('Episode: {} Reward: {}'.format(e+1, mean))
        print('Done run with args: {}'.format(self.args))
        self._save_log()

        self.agent.eval_mode = eval_mode

    def evaluate(self, n_episodes):
        eval_mode = self.agent.eval_mode
        self.agent.eval_mode = False

        total_steps = 0
        for e in range(n_episodes):
            total_steps += self._run_one_episode()
            if (e+1) % self.display_frequency == 0:
                mean = np.mean(np.array(self.l_rewards))
                print('Episode: {} Reward: {}'.format(e+1, mean))
        print('Done run with args: {}'.format(self.args))
        self._save_log()

        self.agent.eval_mode = eval_mode

