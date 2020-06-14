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
import copy

from joblib import Parallel, delayed
import multiprocessing


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
        self.GENERATIONS = 20
        self.MAX_STEPS = 1000
        self.num_episodes = 20
        self.population_size = 200
        self.elite_proportion = 0.1

        self.lr_mean = 0.1

        self.spawner = Spawner(self.population_size,
                               self.STATE_SPACE,
                               self.ACTION_SPACE)

        self.populations = defaultdict()
        self.population_elites = defaultdict()
        self.population_performance = defaultdict()
        self.means = []
        self.covs = []

    def run_experiment(self):


        self.populations[0] = self.spawner.generate_initial_population()

        current_means = self.calculate_population_means(0)
        # current_cov = self.calculate_population_covariances(0, current_means)

        for gen_idx in range(self.GENERATIONS):
            self.means.append(current_means)
            # self.covs.append(current_cov)
            self.evaluate_one_generation(gen_idx)
            self.select_top_performers(gen_idx, self.elite_proportion)
            next_means = self.calculate_next_means(gen_idx, current_means)
            self.populations[gen_idx + 1] = self.spawner.generate_population(next_means, 0)
            # next_cov = self.calculate_population_covariances(gen_idx + 1, next_means)
            # print("=========================")
            # print(current_means)
            # print("--------------------")
            # print(next_means)
            current_means = next_means
            # print(self.average_elite_performance(gen_idx))
            print('\rGeneration {}\tAverage Elite Score: {:.2f}\tAverage Whole Population Score: {:.2f}'.format(gen_idx, self.average_elite_performance(gen_idx), self.average_whole_performance(gen_idx)))
            # print(self.means)

    def average_elite_performance(self, gen_idx):
        elites = self.population_elites[gen_idx]
        return np.sum([x[1] for x in elites]) / len(elites)

    def average_whole_performance(self, gen_idx):
        population = self.population_performance[gen_idx]
        # print(population.values())
        return np.sum(list(population.values())) / len(population)

    def calculate_next_means(self, gen_idx, current_means):
        next_means = copy.deepcopy(current_means)
        mean_difference = self._prep_base_state_dict()
        for layer in mean_difference:
            current_elites = self.population_elites[gen_idx]
            for member_idx, _ in current_elites:
                current_member_state = self.populations[gen_idx][member_idx].state_dict()
                mean_difference[layer] += (current_member_state[layer] - current_means[layer])
            next_means[layer] += (self.lr_mean * mean_difference[layer])
        # print("=========================")
        # print(current_means)
        # print("--------------------")
        # print(next_means)
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
        reward_window = deque(maxlen=30)
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
