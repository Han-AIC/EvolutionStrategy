import sys
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict, Counter
import matplotlib.pyplot as plt
import copy
import time
import gc
import json

from spawn import Spawner
from environment import Environment
from collections import defaultdict

class EvoStrat_Experiment:

    def __init__(self, env_name):

        self.STATE_SPACE = gym.make(env_name).observation_space.shape[0]
        self.ACTION_SPACE = gym.make(env_name).action_space.n
        self.GENERATIONS = 1
        self.MAX_STEPS = 500
        self.num_episodes = 100
        self.num_progenitors = 5
        self.population_size = 5

        self.spawner = Spawner(self.num_progenitors,
                               self.STATE_SPACE,
                               self.ACTION_SPACE)

        self.progenitor_mean_sigma = defaultdict()
        self.populations = defaultdict()

        self._declare_initial_progenitor_means()

    def generate_population(self, gen_idx):
        self.populations[gen_idx] = defaultdict()
        for progenitor in self.progenitor_mean_sigma[gen_idx]:
            mean = self.progenitor_mean_sigma[gen_idx][progenitor][0]
            std = self.progenitor_mean_sigma[gen_idx][progenitor][1]
            population = self.spawner.generate_population(mean, std)
            self.populations[gen_idx][progenitor] = population

    def _declare_initial_progenitor_means(self):
        self._bin_param_space(0, 0.1)

    def _bin_param_space(self, gen_idx, step_size):
        self.progenitor_mean_sigma[gen_idx] = defaultdict()
        for i in range(1, self.num_progenitors + 1):
            mean = -1 + ((2/(self.num_progenitors + 1)) * i)
            self.progenitor_mean_sigma[gen_idx][i - 1] = (mean, step_size)

    def return_progenitors_mean_sigma(self):
        return self.progenitor_mean_sigma

    def return_populations(self):
        return self.populations

    # def run_one_generation(self):
    #     for progenitor in progenitors:
    #         environment = Environment(progenitors[progenitor]["model"])
    #         reward_window = deque(maxlen=20)
    #         for episode_idx in range(1, num_episodes+1):
    #             print("\rEpisode {} of {}".format(episode_idx, num_episodes), end="")
    #             state = environment.reset()
    #             action = environment.select_action_from_policy(state)
    #             reward_per_episode = 0
    #             for i in range(MAX_STEPS):
    #                 next_state, reward, done, info = environment.step(action)
    #                 next_action = environment.select_action_from_policy(next_state)
    #                 action = next_action
    #                 state = next_state
    #                 reward_per_episode += reward
    #                 if done:
    #                     reward_window.append(reward_per_episode)
    #                     break
    #             if episode_idx % 10 == 0:
    #                 print("\rAverage reward: {} by episode: {}".format(np.mean(reward_window), episode_idx))
    #         progenitors[progenitor].update({"average_score": np.mean(reward_window)})

    # for gen_idx in range(GENERATIONS):
    #     for progenitor in progenitors:
    #         environment = Environment(progenitors[progenitor]["model"])
    #         reward_window = deque(maxlen=20)
    #         for episode_idx in range(1, num_episodes+1):
    #             print("\rEpisode {} of {}".format(episode_idx, num_episodes), end="")
    #             state = environment.reset()
    #             action = environment.select_action_from_policy(state)
    #             reward_per_episode = 0
    #             for i in range(MAX_STEPS):
    #                 next_state, reward, done, info = environment.step(action)
    #                 next_action = environment.select_action_from_policy(next_state)
    #                 action = next_action
    #                 state = next_state
    #                 reward_per_episode += reward
    #                 if done:
    #                     reward_window.append(reward_per_episode)
    #                     break
    #             if episode_idx % 10 == 0:
    #                 print("\rAverage reward: {} by episode: {}".format(np.mean(reward_window), episode_idx))
    #         progenitors[progenitor].update({"average_score": np.mean(reward_window)})
    #
    # print(progenitors)
