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

MAX_STEPS = 500
num_episodes = 100

env = gym.make('CartPole-v1')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

STATE_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

population_spawner = Spawner(1, STATE_SPACE, ACTION_SPACE)

progenitors = population_spawner.generate_progenitor_population()
print(progenitors)

# for episode_idx in range(1, num_episodes+1):
#     print("\rEpisode {} of {}".format(episode_idx, num_episodes), end="")
#
#     state = env.reset()
#     action = select_action_from_policy(state, policy_model)
#
#     reward_per_episode = 0
#     for i in range(MAX_STEPS):
#         next_state, reward, done, info = env.step(action)
#         next_action = select_action_from_policy(next_state, policy_model)
#         action = next_action
#         state = next_state
#         reward_per_episode += reward
#         if done:
#             reward_window.append(reward_per_episode)
#             break
#     if episode_idx % 10 == 0:
#         print("\rAverage reward: {} by episode: {}".format(np.mean(reward_window), episode_idx))
