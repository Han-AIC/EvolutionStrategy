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

from experiment import EvoStrat_Experiment

env_name = 'CartPole-v1'
experiment = EvoStrat_Experiment(env_name)

# print(experiment.return_progenitors_mean_sigma())

# experiment.generate_first_population()

# print(experiment.return_populations()[0][0][0])
# experiment.generate_population(0)
# print(experiment.return_populations())

# print(experiment.return_progenitors()[0][0].state_dict())

# print(experiment.return_populations()[0][4][1].state_dict())

means = experiment.calculate_population_means(0, 0)

cov = experiment.calculate_population_covariances(0, 0, means)
print(cov)

# for layer in experiment.return_populations()[0][0][0].state_dict():
#     if layer.split('.')[1] == 'weight':
#         shape = experiment.return_populations()[0][0][0].state_dict()[layer].shape
#         sampled_weights = np.random.normal(-0.6, 0.1, shape)
#         print(torch.from_numpy(sampled_weights).shape)
# print('-------')
# print(experiment.return_progenitors()[0][1].state_dict())
# print(2/6)
# print(np.arange(-1+(2/6), 1, (2/6)))


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
