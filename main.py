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
experiment.run_experiment()
