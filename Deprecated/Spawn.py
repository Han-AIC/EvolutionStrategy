import random
import numpy as np
from collections import defaultdict, namedtuple


class Spawner:
    '''
        Consider making use of Gaussian prior to inform the select of progenitors.
        Consider using Spawner to introduce new individuals to a population at
        each stage: analogous to individuals traveling from foreign populations.
        May prevent stagnancy?

        The MetaComponent Class selects values for the following hyperparameters.
        Each sampling comes from a np.random.choice over [-1, 0, 1] with adjustable
        distribution.

        Learning Rate: {1: 0.0001,
                        2: 0.001,
                        3: 0.01,
                        4: 0.1}

        Choice of Optimizer: {1: RMSProp,
                              2: SGD,
                              3: Adam,
                              4: Adagrad}

        Number of Layers: {1: 1,
                           2: 2,
                           3: 3,
                           4: 4,
                           5: 5,
                           6: 6}

        Num Components per Layer: {1: 1,
                                   2: 2,
                                   3: 3}

        Model Component: {1: 8,
                          2: 16,
                          3: 32,
                          4: 64,
                          5: 128}

        Activation per Component: {1: relu,
                                   2: sigmoid}

        Dropout Level: {1: 0.01,
                        2: 0.1,
                        3: 0.2,
                        4: 0.3,
                        5: 0.4,
                        6: 0.5,
                        7: 0.6,
                        8: 0.7,
                        9: 0.8,
                        10: 0.9}

        Additional?

        Component Size Modifier, Batchnorms on random subsets
    '''
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        self.INPUT_DIM = INPUT_DIM
        self.OUTPUT_DIM = OUTPUT_DIM

    def spawn_single_progenitor(self):
        self._reset_random_seed()

        gene = defaultdict()
        layers = defaultdict()

        LR = self._learning_rate()
        optimizer = self._optimizer()
        num_layers = self._num_layers()

        for layer_idx in range(num_layers):
            num_components = self._num_components_per_layer()
            layers[str(layer_idx)] = {"num_components": num_components,
                                      "components": []}
            for component_idx in range(num_components):
                component = self._select_component_size()
                activation = self._select_activation()
                dropout_p = self._dropout_p_per_component()
                layers[str(layer_idx)]["components"].append((component, activation))

        gene["learning_rate"] = LR
        gene["optimizer"] = optimizer
        gene["num_layers"] = num_layers
        gene["layers"] = layers

        return gene

    def _reset_random_seed(self):
        np.random.seed(random.randint(1, 99999))

    def _learning_rate(self,
                       probs=[0.25, 0.25, 0.25, 0.25]):
        return np.random.choice([1, 2, 3, 4],
                                p=probs)

    def _optimizer(self,
                   probs=[0.25, 0.25, 0.25, 0.25]):
        return np.random.choice([1, 2, 3, 4],
                                p=probs)

    def _num_layers(self,
                    probs=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
        return np.random.choice([1, 2, 3, 4, 5, 6],
                                p=probs)

    def _num_components_per_layer(self,
                                  probs=[0.25, 0.25, 0.25, 0.25]):
        return np.random.choice([1, 2, 3, 4],
                                p=probs)

    def _dropout_p_per_component(self,
                                 probs=[1/14, 1/14,
                                        1/14, 1/14,
                                        1/14, 1/14,
                                        1/14, 1/14,
                                        1/14, 1/14,
                                        1/14, 1/14,
                                        1/14, 1/14]):
        return np.random.choice([1, 2,
                                 3, 4,
                                 5, 6,
                                 7, 8,
                                 9, 10,
                                 11, 12,
                                 13, 14],
                                 p=probs)

    def _select_component_size(self,
                          probs=[0.2, 0.2, 0.2, 0.2, 0.2]):
        return np.random.choice([1, 2, 3, 4, 5],
                                p=probs)

    def _select_activation(self,
                          probs=[0.5, 0.5]):
        return np.random.choice([1, 2],
                                p=probs)
