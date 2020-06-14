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
                                                            "out_features": 8},
                                   "layer_type": "linear",
                                   "activation": "relu"},
                                "1":{"layer_size_mapping": {"in_features": 8,
                                                           "out_features": output_dim},
                                   "layer_type": "linear",
                                   "activation": "nil"}}

    def generate_initial_population(self):
        population = {}
        for i in range(self.pop_size):
            member = Model(self.model_structure)
            # member = self.resample_member_state_dict(member, mean, step_size)
            population.update({i: member})
        return population

    # def resample_member_state_dict(self, member, mean, step_size):
    #     state_dict = member.state_dict()
    #     for layer in state_dict:
    #         shape = state_dict[layer].shape
    #         sampled_weights = torch.from_numpy(np.random.normal(mean, step_size, shape))
    #         state_dict.update({layer: sampled_weights})
    #     member.load_state_dict(state_dict)
    #     return member
