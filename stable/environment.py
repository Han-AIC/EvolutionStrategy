import torch
import random
import numpy as np
import gym

class Environment:

    """
    1. Instantiates an environment for each population member to undergo evaluation.
    2. Keeps track of states over time, selects actions probabilistically from the
       output of each member model.
    3. Steps environment forward using selected action.
    4. Resets environment using a new novel random seed.
    """

    def __init__(self, model):
        """
        Attributes:
                    model (nn.module): A PyTorch neural network.
                    env (nn.module): Gym environment.
                    current_state (tuple): The current state of the environment.
                    state_shape (tuple): The shape of observations from the environment.
                    action_shape (int): The number of legitimate actions in env.
        """
        self.model = model
        self.env = gym.make('CartPole-v1')
        self.current_state = self.env.reset()
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n

    def select_action_from_policy(self, state):
        """
        Probabilistically selects an action by running the state through a PyTorch model,
        applying a softmax, and using the resulting vector with np.random.choice.
        Inputs:
                state (tuple): An observation from the environment.

        Returns:
                action (int): An action probabilistically selected from self.model.
        """
        state = torch.from_numpy(state).float()
        action_probabilities = torch.softmax(self.model(state), dim = 0)
        action_probabilities = action_probabilities.detach().numpy()
        action = np.random.choice([i for i in range(self.action_shape)],
                                  p=action_probabilities)
        return action

    def reset(self):
        self.env.seed(random.randint(0, 99999))
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
