from model import Model

class Spawner:

    def __init__(self,
                pop_size,
                input_dim,
                output_dim):
        self.pop_size = pop_size
        self.model_structure = {"0":{"layer_size_mapping": {"in_features": input_dim,
                                                            "out_features": 32},
                                   "layer_type": "linear",
                                   "activation": "relu"},
                                "1":{"layer_size_mapping": {"in_features": 32,
                                                           "out_features": output_dim},
                                   "layer_type": "linear",
                                   "activation": "nil"}}

    def generate_initial_progenitors(self):
        population = {}
        for i in range(self.pop_size):
            single_progenitor = Model(self.model_structure)
            population.update({i: single_progenitor})
        return population
