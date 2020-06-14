import torch
import numpy as np
from model import Model

class Spawner:

    def __init__(self,
                pop_size,
                input_dim,
                output_dim):
        self.pop_size = pop_size
        self.model_structure = {"0":{"layer_size_mapping": {"in_features": input_dim,
                                                            "out_features": 16},
                                   "layer_type": "linear",
                                   "activation": "relu"},
                                "1":{"layer_size_mapping": {"in_features": 16,
                                                           "out_features": output_dim},
                                   "layer_type": "linear",
                                   "activation": "nil"}}
                                   
    def update_pop_size(self, population_size):
        self.pop_size = population_size

    def generate_initial_population(self):
        population = {}
        for i in range(self.pop_size):
            member = Model(self.model_structure)
            population.update({i: member})
        return population

    def generate_population(self, mean, step_size):
        population = {}
        for i in range(self.pop_size):
            member = Model(self.model_structure)
            sampled_state_dict = self.resample_member_state_dict(member, mean, step_size)
            member.load_state_dict(sampled_state_dict)
            population.update({i: member})
        return population

    def resample_member_state_dict(self, member, mean, step_size):
        state_dict = member.state_dict()
        for layer in state_dict:
            shape = state_dict[layer].shape
            base = np.zeros(shape)
            if layer.split('.')[1] == 'weight':
                for i, param_arr in enumerate(mean[layer]):
                    for j, param in enumerate(mean[layer][i]):
                        base[i][j] = np.random.normal(mean[layer][i][j], step_size)
            else:
                for i, param in enumerate(mean[layer]):
                    base[i] = np.random.normal(mean[layer][i], step_size)
            state_dict[layer] = torch.from_numpy(base)
        return state_dict
