from model import Model

class Spawner:

    def __init__(self,
                pop_size,
                input_dim,
                output_dim):
        self.pop_size = pop_size
        self.model_structure = {"0":{"layer_size_mapping": {"in_features": input_dim,
                                                            "out_features": 64},
                                   "layer_type": "linear",
                                   "activation": "relu"},
                                "1":{"layer_size_mapping": {"in_features": 64,
                                                           "out_features": output_dim},
                                   "layer_type": "linear",
                                   "activation": "nil"}}

    def generate_progenitor_population(self):
        population = {}
        for i in range(self.pop_size):
            individual = {"model": Model(self.model_structure),
                          "average_score": 0}
            population.update({i: individual})
        return population
