import logging
import os
import pickle
from collections import deque
import time
from copy import deepcopy
import numpy as np

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
floatX = np.float32


def plot(data={}, loc="visualization.pdf", x_label="", y_label="", title="", kind='line',
         legend=True, index_col=None, clip=None, moving_average=False):
    if all([len(data[key]) > 1 for key in data]):
        if moving_average:
            smoothed_data = {}
            for key in data:
                smooth_scores = [np.mean(data[key][max(0, i - 10):i + 1]) for i in range(len(data[key]))]
                smoothed_data['smoothed_' + key] = smooth_scores
                smoothed_data[key] = data[key]
            data = smoothed_data
        df = pd.DataFrame(data=data)
        ax = df.plot(kind=kind, legend=legend, ylim=clip)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(loc)
        plt.close()


def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)


class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


class Q_Network(object):
    def __init__(self, state_shape, nb_actions, init_q, gamma, alpha, learning_method, rng, freeze=False):
        self.state_shape = list(state_shape)
        self.nb_actions = nb_actions
        self.freeze = freeze  # no learning if True
        self.init_q = init_q
        self.q = dict()
        self.gamma = gamma
        self.alpha = alpha
        self.start_alpha = deepcopy(alpha)
        self.learning_method = learning_method
        self.rng = rng

    def reset(self):
        self.q = dict()
        self.alpha = deepcopy(self.start_alpha)

    def get_q(self, s, a):
        sa = tuple(list(s) + [a])
        if sa in self.q:
            return self.q[sa]
        else:
            self._set_init_q(s)
            return self.init_q

    def _set_q(self, s, a, q):
        sa = tuple(list(s) + [a])
        if sa not in self.q:
            self._set_init_q(s)
        self.q[sa] = floatX(q)

    def _set_init_q(self, s):
        s = list(s)
        for a in range(self.nb_actions):
            self.q[tuple(s + [a])] = floatX(self.init_q)

    def get_max_action(self, s, stochastic=True):
        values = np.asarray([self.get_q(s, a) for a in range(self.nb_actions)])
        if stochastic:
            actions = np.where(values == values.max())[0]
            return self.rng.choice(actions)
        else:
            return np.argmax(values)

    def learn(self, s, a, r, s2, term):
        if self.freeze:
            return
        if self.learning_method == 'ql':
            self._ql(s, a, r, s2, term)
        else:
            raise ValueError('Learning method is not recognised.')

    def _ql(self, s, a, r, s2, term):
        if term:
            q2 = 0.
        else:
            values = np.asarray([self.get_q(s2, a) for a in range(self.nb_actions)])
            q2 = np.max(values)
        delta = r + self.gamma * q2 - self.get_q(s, a)
        self._set_q(s, a, self.get_q(s, a) + self.alpha * delta)  # updating Q


class QNetCount(Q_Network):
    """
    Similar to Q_Network, but with (s,a)-visit-counter capability
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_count = dict()

    def reset(self):
        super().reset()
        self.n_count = dict()

    def get_count(self, s, a):
        sa = tuple(list(s) + [a])
        if sa in self.n_count:
            return self.n_count[sa]
        else:
            self._init_count(s)
            return np.int32(0)

    def increment_count(self, s, a):
        sa = tuple(list(s) + [a])
        if sa not in self.n_count:
            self._init_count(s)
        self.n_count[sa] += 1

    def _init_count(self, s):
        s = list(s)
        for a in range(self.nb_actions):
            self.n_count[tuple(s + [a])] = np.int32(0)


class Experiment(object):
    def __init__(self, qnet, qnet_explore, Q_R, env, saving_period, printing_period, writing_period, epsilon, final_epsilon,
                 annealing_episodes, annealing, annealing_start_episode, learning_method, explore_method, rng,
                 episode_max_len, make_folder=True, folder_location='/results/', folder_name='tabular'):
        self.qnet = qnet
        self.qnet_explore = qnet_explore
        self.qr = Q_R
        self.rng = rng
        self.env = env
        self.episode_max_len = episode_max_len
        self.learning_method = learning_method
        self.explore_method = explore_method
        if explore_method in ['count', 'secure_count']:
            assert isinstance(self.qnet, QNetCount)
        self.last_state = None
        self.action = None
        self.epsilon = epsilon
        self.init_epsilon = deepcopy(epsilon)
        self.final_epsilon = final_epsilon
        self.annealing_start_episode = annealing_start_episode
        self.annealing_episodes = annealing_episodes
        self.annealing = annealing
        self.score_agent = 0
        self.step_in_episode = 0
        self.total_learning_steps = 0  # is not reset
        self.saving_period = saving_period
        self.printing_period = printing_period
        self.writing_period = writing_period
        self.episode_done = False
        if make_folder:
            self.folder_name = self._create_folder(folder_location, folder_name)
        else:
            self.folder_name = self._get_folder(folder_location, folder_name)
        self.model = {}
        self.convergence_count = 0
        self.count_episode = 0
        self.annealing_amount = (self.init_epsilon - self.final_epsilon) / (self.annealing_episodes + 1)
        self.actions_hist = None
        self.reset()

    def reset(self):
        self.last_state = self.env.reset()
        self.action = self._get_action(self.last_state, explore=True)
        self.score_agent = 0

    def print_q(self):
        print()
        print(" " * 5, end="")
        for s in range(self.env.bridge_len):
            print(" {:2}      ".format(s), end="")
        print()
        print("   ┌" + ("─" * (9 * self.env.bridge_len)))
        for a in range(self.env.nb_actions):
            print("{:2} │ ".format(a), end="")
            for s in range(self.env.bridge_len):
                print("{:6.2f} │ ".format(self.qnet.get_q([s, 0, 0], a)), end="")
            print()
        print()

    def run(self, nb_episodes, learning=True, target_eval=True, nb_eval=1, print_q=False):
        """
        If target_eval is True, target policy is evaluated; otherwise, training eval (with exploration) is reported
        """
        eval_returns = []
        eval_steps = []
        a0 = []
        a1 = []
        a2 = []
        eval_returns_window = []
        eval_steps_window = []
        a0_window = []
        a1_window = []
        a2_window = []
        for count_episode in range(nb_episodes + 1):
            self.count_episode = count_episode
            self.episode_done = False
            while not self.episode_done:
                rt = self._do_episode(is_learning=learning)
                if print_q and count_episode % self.printing_period == 0:
                    self.print_q()
                if target_eval:  # greedy evaluation of target policy
                    rt = []
                    step_in_episode = []
                    for _ in range(nb_eval):  # evaluation episodes
                        self.episode_done = False
                        rt.append(self._do_episode(is_learning=False))
                        step_in_episode.append(self.step_in_episode)
                    rt = np.mean(rt)
                    self.step_in_episode = np.mean(step_in_episode)
                eval_returns_window.append(deepcopy(rt))
                eval_steps_window.append(deepcopy(self.step_in_episode))
                a0_window.append(self.actions_hist[0])
                a1_window.append(self.actions_hist[1])
                a2_window.append(self.actions_hist[2])
                if count_episode % self.writing_period == 0:
                    eval_returns.append(float(np.mean(eval_returns_window)))
                    eval_steps.append(float(np.mean(eval_steps_window)))
                    a0.append(float(np.mean(a0_window)))
                    a1.append(float(np.mean(a1_window)))
                    a2.append(float(np.mean(a2_window)))
                    eval_returns_window = []
                    eval_steps_window = []
                    a0_window = []
                    a1_window = []
                    a2_window = []
                if count_episode % self.printing_period == 0:
                    print(Font.bold + Font.yellow + ' >>> Episode: {}'.format(count_episode) + Font.end +
                          ' | Steps: {0} | Score: {1}'.format(self.total_learning_steps, rt) + Font.end)
                if learning and count_episode % self.saving_period == 0:
                    try:
                        self._plot_and_write(plot_dict={'scores': eval_returns}, loc=self.folder_name + "/scores",
                                             x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                             moving_average=True)
                        self._plot_and_write(plot_dict={'steps': eval_steps}, loc=self.folder_name + "/steps",
                                             x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)
                        self._plot_and_write(plot_dict={'action 0': a0}, loc=self.folder_name + "/a0",
                                             x_label="Eps.", y_label="Mean # of a0", title="", kind='line', legend=True)
                        self._plot_and_write(plot_dict={'action 1': a1}, loc=self.folder_name + "/a1",
                                             x_label="Eps.", y_label="Mean # of a1", title="", kind='line', legend=True)
                        self._plot_and_write(plot_dict={'action 2': a2}, loc=self.folder_name + "/a2",
                                             x_label="Eps.", y_label="Mean # of a2", title="", kind='line', legend=True)

                        with open(self.folder_name + "/tabular_qnet.pkl", 'wb') as f:
                            pickle.dump(self.qnet, f)
                        with open(self.folder_name + "/tabular_qd.pkl", 'wb') as f:
                            pickle.dump(self.qnet_explore, f)
                        with open(self.folder_name + "/tabular_qr.pkl", 'wb') as f:
                            pickle.dump(self.qr, f)
                        with open(self.folder_name + "/tabular_experiment.pkl", 'wb') as f:
                            pickle.dump(self, f)
                    except:  # bypassing opened files in Windows
                        pass
        return eval_returns

    def _do_episode(self, is_learning):
        self.step_in_episode = 0
        self.actions_hist = [0] * self.env.nb_actions
        self.reset()
        if is_learning and self.learning_method == 'fw_sarsa':
            self.qnet.new_episode()
        episode_return = 0
        while not self.episode_done:
            r = self._step(is_learning=is_learning)
            episode_return += r
            if not self.episode_done and self.step_in_episode > self.episode_max_len:
                logger.warning('Reaching maximum number of steps in the current episode.')
                self.episode_done = True
        if is_learning and self.learning_method == 'fw_sarsa':
            self.qnet.end_of_episode()
        if self.annealing and is_learning:
            self._anneal()
        return episode_return

    def _get_action(self, s, explore):
        if self.explore_method == 'secure':
            return self._get_secure_e_greedy_action(s, explore)
        elif self.explore_method == 'egreedy':
            return self._get_e_greedy_action(s, explore)
        elif self.explore_method == 'softmax':
            return self._get_softmax_action(s, explore)
        elif self.explore_method == 'count':
            return self._get_count_based_action(s, explore)
        else:
            raise ValueError('Exploration method is not defined.')

    def _get_e_greedy_action(self, s, explore):
        if explore and self.rng.binomial(1, self.epsilon):
            action = self.rng.randint(self.env.nb_actions)
        else:
            action = self.qnet.get_max_action(s)
        return action

    @staticmethod
    def softmax(x):
        x = np.asanyarray(x, dtype=np.float64)
        x -= np.max(x)  # std technique to prevent overflow
        np.exp(x, x)
        x /= np.sum(x)
        # assuring stability
        x[x > 1] = 1
        x[x < 1e-6] = 0
        return x

    def _get_softmax_action(self, s, explore):
        # uses AI policy
        if explore:
            q = np.asarray([self.qnet.get_q(s, a) for a in range(self.env.nb_actions)], dtype=np.float64)
            q = self.softmax(q)
            selector = self.rng.multinomial(1, q)
            action = int(np.where(selector == 1)[0])
        else:
            action = self.qnet.get_max_action(s)
        return action

    def _get_count_based_action(self, s, explore):
        # uses AI policy
        if explore:
            c = np.asarray([self.qnet.get_count(s, a) for a in range(self.env.nb_actions)], dtype=np.float64)
            q = np.asarray([self.qnet.get_q(s, a) for a in range(self.env.nb_actions)], dtype=np.float64)
            q = q + (1.0 / (1.0 + c ** 0.5))  # plus count-based motivation
            actions = np.where(q == q.max())[0]
            action = self.rng.choice(actions)
        else:
            action = self.qnet.get_max_action(s)
        return action

    def _get_secure_uniform_action(self, s):
        # Uniform and secure
        q = np.asarray([self.qnet_explore.get_q(s, a) for a in range(self.env.nb_actions)], dtype=np.float64)
        if all(abs(q + 1) < 1e-5):  # if all values are -1
            return self.rng.randint(0, self.env.nb_actions)
        else:
            eta = (1.0 + q) / (self.env.nb_actions + np.sum(q))
            selector = self.rng.multinomial(1, eta)
            return int(np.where(selector == 1)[0])

    def _get_secure_e_greedy_action(self, s, explore):
        if explore and self.rng.binomial(1, self.epsilon):
            action = self._get_secure_uniform_action(s)
        else:
            action = self.qnet.get_max_action(s)
        return action

    def _get_secure_action_threshold(self, s, q_threshold):
        # Implementation of Theorem 2 of the paper (not used in the examples of the paper though).
        q = np.asarray([self.qnet.get_q(s, a) for a in range(self.env.nb_actions)])
        q = q > q_threshold
        actions = np.where(q == True)[0]
        if len(actions) == 0:
            action = self.rng.randint(0, self.env.nb_actions)
        else:
            action = int(self.rng.choice(actions))
        return action, list(actions)

    def _step(self, is_learning=True):
        if self.learning_method == 'ql':
            r = self._step_ql(is_learning=is_learning)
        else:
            raise NotImplementedError(self.learning_method)
        return r

    def _step_ql(self, is_learning=True):
        action = self._get_action(self.last_state, is_learning)
        self.actions_hist[action] += 1
        if is_learning and self.explore_method in ['count', 'secure_count']:
            self.qnet.increment_count(self.last_state, action)
        s2, r_env, self.episode_done, _ = self.env.step(action)
        if r_env != 0:
            term = True
        else:
            term = self.episode_done
        # rewards:
        term_ex = False
        if r_env != 0:
            # uncomment to use neg reward for baselines
            # if self.explore_method not in ['secure', 'secure_count']:
            #     r_tr = -1
            term_ex = True

        if is_learning:
            self.qnet.learn(self.last_state, action, r_env, s2, term)
            # if self.explore_method in ['secure', 'secure_count']:
            self.qnet_explore.learn(self.last_state, action, min(r_env, 0.0), s2, term_ex) # Reward will be 0.0 unless we experience a 'death' transition
            self.qr.learn(self.last_state, action, max(0, r_env), s2, term_ex)           # Reward will be 0.0 unless we experience a 'recovery' transition

        self.last_state = deepcopy(s2)
        self.score_agent += r_env
        self.step_in_episode += 1
        if is_learning:
            self.total_learning_steps += 1
        return r_env

    def _anneal(self):
        if self.count_episode > self.annealing_start_episode:
            if self.count_episode < self.annealing_start_episode + self.annealing_episodes - 1:
                self.epsilon -= self.annealing_amount
            else:
                self.epsilon = self.final_epsilon

    @staticmethod
    def _plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                        moving_average=False):
        for key in plot_dict:
            plot(data={key: plot_dict[key]}, loc=loc + ".pdf", x_label=x_label, y_label=y_label, title=title,
                 kind=kind, legend=legend, index_col=None, moving_average=moving_average)
            write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")

    @staticmethod
    def _create_folder(folder_location, folder_name):
        i = 0
        while os.path.exists(os.getcwd() + folder_location + folder_name + str(i)):
            i += 1
        folder_name = os.getcwd() + folder_location + folder_name + str(i)
        os.mkdir(folder_name)
        return folder_name

    @staticmethod
    def _get_folder(folder_location, folder_name):
        i = 0
        while os.path.exists(os.getcwd() + folder_location + folder_name + str(i)):
            i += 1
        return os.getcwd() + folder_location + folder_name + str(i - 1)

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
