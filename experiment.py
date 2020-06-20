import os
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

from tqdm import tqdm, tnrange, tqdm_notebook

class EvoStrat_Experiment:

    """
    1. Instantiates an environment for each population member to undergo evaluation.
    2. Keeps track of states over time, selects actions probabilistically from the
       output of each member model.
    3. Steps environment forward using selected action.
    4. Resets environment using a new novel random seed.
    """

    def __init__(self,
                 experiment_params):

        self.env_name = experiment_params['env_name']
        self.model_structure = experiment_params['model_structure']

        self.STATE_SPACE = gym.make(self.env_name).observation_space.shape[0]
        self.ACTION_SPACE = gym.make(self.env_name).action_space.n
        self.GENERATIONS = experiment_params['GENERATIONS']
        self.MAX_STEPS = experiment_params['MAX_STEPS']
        self.num_episodes = experiment_params['num_episodes']
        self.population_size = experiment_params['population_size']
        self.population_size_decay = experiment_params['population_size_decay']
        self.minimum_population_size = experiment_params['minimum_population_size']
        self.elite_proportion = experiment_params['elite_proportion']
        self.step_size = experiment_params['step_size']
        self.step_size_decay = experiment_params['step_size_decay']
        self.minimum_step_size = experiment_params['minimum_step_size']
        self.lr_mean = experiment_params['lr_mean']

        self.spawner = Spawner(self.population_size,
                               self.STATE_SPACE,
                               self.ACTION_SPACE,
                               self.model_structure)

        self.populations = defaultdict()
        self.population_elites = defaultdict()
        self.population_performance = defaultdict()
        self.means = []

        self.best_performer = {'gen_idx': 0,
                                'member_idx': 0,
                                'performance': 0}

    def run_experiment(self):
        pbar_overall = tqdm(total=self.GENERATIONS, desc='Generations', leave=True)

        self.populations[0] = self.spawner.generate_initial_population()
        current_means = self.calculate_population_means(0)
        for gen_idx in range(self.GENERATIONS):
            self.means.append(current_means)
            self.evaluate_one_generation(gen_idx)
            self.select_top_performers(gen_idx, self.elite_proportion)
            next_means = self.calculate_next_means(gen_idx, current_means)

            step_size = max(self.step_size, self.minimum_step_size)
            self.step_size *= self.step_size_decay

            self.populations[gen_idx + 1] = self.spawner.generate_population(next_means, step_size)

            population_size = max(self.population_size, self.minimum_population_size)
            self.population_size = int(self.population_size * self.population_size_decay)
            self.spawner.update_pop_size(self.population_size)

            current_means = next_means
            print('\rGeneration {}\tAverage Elite Score: {:.2f}\tAverage Whole Population Score: {:.2f}'.format(gen_idx, self.average_elite_performance(gen_idx), self.average_whole_performance(gen_idx)))
            self.cull_non_elite(gen_idx)
            self.select_best_performer(gen_idx)

            pbar_overall.update(1)
            print("========================")

        self.save_best_performer()
        return str(int(self.best_performer['performance']))

    def select_best_performer(self, gen_idx):
        for elite in self.population_elites[gen_idx]:
            if elite[1] > self.best_performer['performance']:
                self.best_performer = {'gen_idx': gen_idx,
                                        'member_idx': elite[0],
                                        'performance': elite[1]}

    def save_best_performer(self):
        print("========================")
        print("Best Performer")
        print(self.best_performer)
        model_state = self.populations[self.best_performer['gen_idx']][self.best_performer['member_idx']].state_dict()
        SOLUTION_PATH = './solutions/' + self.env_name + '_' + str(int(self.best_performer['performance']))
        if not os.path.exists(SOLUTION_PATH):
        	os.makedirs(SOLUTION_PATH)
        torch.save(model_state, SOLUTION_PATH + '/solution.pth')

    def cull_non_elite(self, gen_idx):
        elite_indices = [x[0] for x in self.population_elites[gen_idx]]
        self.populations[gen_idx] = {idx:model for idx, model in self.populations[gen_idx].items() if idx in elite_indices}
        self.population_performance[gen_idx] = {idx:performance for idx, performance in self.population_performance[gen_idx].items() if idx in elite_indices}

    def average_elite_performance(self, gen_idx):
        elites = self.population_elites[gen_idx]
        return np.sum([x[1] for x in elites]) / len(elites)

    def average_whole_performance(self, gen_idx):
        population = self.population_performance[gen_idx]
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
        pbar_within_generation = tqdm(total=len(self.populations[gen_idx].keys()), desc='Members', leave=False)
        for member_idx in self.populations[gen_idx]:
            self.evaluate_one_member(gen_idx,
                                     member_idx)
            pbar_within_generation.update(1)

    def evaluate_one_member(self,
                            gen_idx,
                            member_idx):
        current_member = self.populations[gen_idx][member_idx]
        environment = Environment(current_member, self.env_name)
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

    def _prep_base_state_dict(self):
        template_idx = list(self.populations[0].keys())[0]
        base_state_dict = self.populations[0][template_idx].state_dict()
        for layer in base_state_dict:
            zeroes = torch.from_numpy(np.zeros(base_state_dict[layer].shape))
            base_state_dict.update({layer: zeroes})
        return base_state_dict
