import json
import numpy as np
from collections import namedtuple, deque, OrderedDict, Counter
import torch
from model import Model
from environment import Environment

ENV_NAME = 'CartPole-v1'
NUM_EPISODES = 20
REWARD_WINDOW_SIZE = 30
SOLUTION_SCORE = 135

solution_path = "./solutions/" + ENV_NAME + '_' + str(SOLUTION_SCORE) + "/solution.pth"
params_path = "./solutions/" + ENV_NAME + '_' + str(SOLUTION_SCORE) + "/model_params.json"

with open(params_path) as json_file:
    experiment_params = json.load(json_file)

MAX_STEPS = experiment_params['MAX_STEPS']
solution_model_structure = experiment_params['model_structure']

solution_model = Model(solution_model_structure)
solution_model.load_state_dict(torch.load(solution_path), strict=False)

environment = Environment(solution_model, ENV_NAME)
reward_window = deque(maxlen=REWARD_WINDOW_SIZE)
for episode_idx in range(1, NUM_EPISODES):
    environment.render_env()
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

print(np.mean(reward_window))
