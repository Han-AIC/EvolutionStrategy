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

GENERATIONS = 1
MAX_STEPS = 500
num_episodes = 100

env = gym.make('CartPole-v1')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

STATE_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

population_spawner = Spawner(1, STATE_SPACE, ACTION_SPACE)

progenitors = population_spawner.generate_progenitor_population()

for gen_idx in range(GENERATIONS):
    for progenitor in progenitors:
        environment = Environment(progenitors[progenitor]["model"])
        reward_window = deque(maxlen=20)
        for episode_idx in range(1, num_episodes+1):
            print("\rEpisode {} of {}".format(episode_idx, num_episodes), end="")
            state = environment.reset()
            action = environment.select_action_from_policy(state)
            reward_per_episode = 0
            for i in range(MAX_STEPS):
                next_state, reward, done, info = environment.step(action)
                next_action = environment.select_action_from_policy(next_state)
                action = next_action
                state = next_state
                reward_per_episode += reward
                if done:
                    reward_window.append(reward_per_episode)
                    break
            if episode_idx % 10 == 0:
                print("\rAverage reward: {} by episode: {}".format(np.mean(reward_window), episode_idx))
        progenitors[progenitor].update({"average_score": np.mean(reward_window)})

print(progenitors)
