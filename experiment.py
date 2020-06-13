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

    """
    1. Instantiates an environment for each population member to undergo evaluation.
    2. Keeps track of states over time, selects actions probabilistically from the
       output of each member model.
    3. Steps environment forward using selected action.
    4. Resets environment using a new novel random seed.
    """

    def __init__(self,
                 env_name):

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

        self._generate_first_population()

    def calculate_population_means(self,
                                   gen_idx,
                                   population_idx):
        mean_state_dict = self._prep_base_state_dict()
        current_population = self.populations[gen_idx][population_idx]
        for member_idx in current_population:
            current_model_state = current_population[member_idx].state_dict()
            for layer in current_model_state:
                mean_state_dict[layer] += current_model_state[layer]
        for layer in mean_state_dict:
            mean_state_dict[layer] /= len(current_population.keys())
        return mean_state_dict

    def calculate_population_covariances(self,
                                       gen_idx,
                                       population_idx,
                                       means):
        covariance_state_dict = self._prep_base_state_dict()
        current_population = self.populations[gen_idx][population_idx]

        for layer in covariance_state_dict:
            if layer.split('.')[1] == 'weight':
                for i, param_arr in enumerate(covariance_state_dict[layer]):
                    for j, param in enumerate(covariance_state_dict[layer][i]):
                        covariance = self.get_covariances(layer,
                                                          current_population,
                                                          means,
                                                          i,
                                                          j)
                        covariance_state_dict[layer][i][j] = covariance
            else:
                for i, param_arr in enumerate(covariance_state_dict[layer]):
                    covariance = self.get_covariances(layer,
                                                      current_population,
                                                      means,
                                                      i)
                    covariance_state_dict[layer][i] = covariance
        return covariance_state_dict

    def get_covariances(self,
                        layer,
                        current_population,
                        means,
                        i,
                        j=None):
        sum = 0
        num_members = len(current_population.keys())
        for member_idx in current_population:
            current_member_state = current_population[member_idx].state_dict()
            if j is not None:
                current_param = current_member_state[layer][i][j].item()
                current_mean = means[layer][i][j].item()
            else:
                current_param = current_member_state[layer][i].item()
                current_mean = means[layer][i].item()
            sum += (current_param - current_mean)**2
        return sum/num_members


    def _prep_base_state_dict(self):
        base_state_dict = self.populations[0][0][0].state_dict()
        for layer in base_state_dict:
            zeroes = torch.from_numpy(np.zeros(base_state_dict[layer].shape))
            base_state_dict.update({layer: zeroes})
        return base_state_dict

    def _generate_first_population(self):
        self._bin_param_space(0, 0.1)
        gen_idx = 0
        self.populations[gen_idx] = defaultdict()
        for progenitor in self.progenitor_mean_sigma[gen_idx]:
            mean = self.progenitor_mean_sigma[gen_idx][progenitor][0]
            std = self.progenitor_mean_sigma[gen_idx][progenitor][1]
            population = self.spawner.generate_population(mean, std)
            self.populations[gen_idx][progenitor] = population

    def _bin_param_space(self,
                         gen_idx,
                         step_size):
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
