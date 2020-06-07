import torch
import random
import numpy as np

class Environment:

    def __init__(self, model):
        self.model = model
        self.env = gym.make('CartPole-v1')
        self.env.seed(random.randint(0, 99999))
        self.current_state = self.env.reset()
        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.n

    def select_action_from_policy(self):
        state = torch.from_numpy(state).float()
        action_probabilities = torch.softmax(self.model(state), dim = 0)
        action_probabilities = action_probabilities.detach().numpy()
        return np.random.choice([0, 1], p=action_probabilities)

    def reset(self):
        self.env.seed(random.randint(0, 99999))
        return env.reset()

    def step(self):
        next_state, reward, done, info = env.step(action)
        return next_state, reward, done, info
