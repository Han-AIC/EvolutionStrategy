import sys
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict, Counter
from operator import itemgetter
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
        self.num_episodes = 40
        self.population_size = 2
        self.mean_learning_rate = 0.5

        self.spawner = Spawner(self.population_size,
                               self.STATE_SPACE,
                               self.ACTION_SPACE)

        self.initial_progenitor_mean_sigma = defaultdict()
        self.populations = defaultdict()
        self.population_elites = defaultdict()
        self.population_performance = defaultdict()
        # self.population_means = defaultdict()
        self.population_covariances = defaultdict()

    def run_experiment(self):

        current_gen = 0
        self.populations[current_gen] = self.spawner.generate_initial_population()
        current_means = self.calculate_population_means(current_gen)
        current_cov = self.calculate_population_covariances(current_gen, current_means)
        self.evaluate_one_generation(current_gen)
        print(self.population_performance)
        self.select_top_performers(0, 0.5)
        print(self.population_elites)
        print("===========")
        print(current_means)
        next_means = self.calculate_next_means(current_gen, current_means)
        print("==========")
        print(next_means)
        # print(self.population_means)

        # print(current_means)

        # self.evaluate_one_generation(0)
        # self.select_top_performers(0, 0.5)

        # current_means = self.calculate_population_means(0, 0)
        # current_cov = self.calculate_population_covariances(0, 0, current_means)
        # self.population_means[0] = defaultdict()
        # self.population_covariances[0] = defaultdict()
        # self.population_means[0][0] = current_means
        # self.population_covariances[0][0] = current_cov
        #
        # print(self.population_means[0][0])
        # self.update_means(0)
        # print("========")
        # print(self.population_means[1][0])


        # for gen_idx in range(0, self.GENERATIONS):
        #     self.population_means[gen_idx] = defaultdict()
        #     self.population_covariances[gen_idx] = defaultdict()
        #
        #     for population_idx in range(len(self.population_means[0].keys()))
        #         self.population_means[gen_idx][population_idx] = current_means
        #         self.population_covariances[gen_idx][population_idx] = current_cov


        # print(self.population_means)


        # for gen_idx in range(self.GENERATIONS):
        #     self.evaluate_one_generation(gen_idx)


        # means = self.calculate_population_means(0, 0)
        #
        # cov = self.calculate_population_covariances(0, 0, means)

        # print(self.progenitor_mean_sigma)
        # print(means)
        # print(cov)

        # for gen_idx in range(self.GENERATIONS):
        #     self.evaluate_one_generation(gen_idx)

    def calculate_next_means(self, gen_idx, current_means):
        next_means = current_means
        mean_difference = self._prep_base_state_dict()
        for layer in mean_difference:
            current_elites = self.population_elites[gen_idx]
            for member_idx, _ in current_elites:
                current_member_state = self.populations[gen_idx][member_idx].state_dict()
                mean_difference[layer] += (current_member_state[layer] - current_means[layer])
            mean_difference[layer] /= len(current_elites)
            next_means[layer] += (self.mean_learning_rate * mean_difference[layer])
        return next_means


    def select_top_performers(self,
                              gen_idx,
                              proportion):
        self.population_elites[gen_idx] = defaultdict()
        current_generation = self.population_performance[gen_idx]
        member_performances = list(current_generation.items())
        sorted_member_performances = sorted(member_performances,
                                       key=itemgetter(1),
                                       reverse=True)
        elites = sorted_member_performances[0:int(len(sorted_member_performances) * proportion)]
        self.population_elites[gen_idx] = elites

    def evaluate_one_generation(self,
                                gen_idx):
        self.population_performance[gen_idx] = defaultdict()
        for member_idx in self.populations[gen_idx]:
            self.evaluate_one_member(gen_idx,
                                     member_idx)

    def evaluate_one_member(self,
                            gen_idx,
                            member_idx):
        current_member = self.populations[gen_idx][member_idx]
        environment = Environment(current_member)
        reward_window = deque(maxlen=10)
        for episode_idx in range(1, self.num_episodes+1):
            state = environment.reset()
            action = environment.select_action_from_policy(state)
            reward_per_episode = 0
            for i in range(self.MAX_STEPS):
                next_state, reward, done, info = environment.step(action)
                next_action = environment.select_action_from_policy(next_state)
                action = next_action
                state = next_state
                reward_per_episode += reward
                if done:
                    reward_window.append(reward_per_episode)
                    break
        self.population_performance[gen_idx][member_idx] = np.mean(reward_window)

    def calculate_population_means(self,
                                   gen_idx):
        mean_state_dict = self._prep_base_state_dict()
        current_population = self.populations[gen_idx]
        for member_idx in current_population:
            current_model_state = current_population[member_idx].state_dict()
            for layer in current_model_state:
                mean_state_dict[layer] += current_model_state[layer]
        for layer in mean_state_dict:
            mean_state_dict[layer] /= len(current_population.keys())
        return mean_state_dict

    def calculate_population_covariances(self,
                                       gen_idx,
                                       means):
        covariance_state_dict = self._prep_base_state_dict()
        current_population = self.populations[gen_idx]
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
        base_state_dict = self.populations[0][0].state_dict()
        for layer in base_state_dict:
            zeroes = torch.from_numpy(np.zeros(base_state_dict[layer].shape))
            base_state_dict.update({layer: zeroes})
        return base_state_dict
