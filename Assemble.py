from collections import defaultdict
import numpy as np

from Model import Model
'''

    Structure of a Gene

        [Meta Component][[Model_Components_1...N], [Model_Components_1...N] ... Layer N]

'''
'''

    Minimum Viable Product

        Produce a population of genes. Meta component restricted to LR, choice
        of optimizer, None or Full BatchNorm, number of layers, number of
        components per layer. Model Components chosen from FC,Conv1D, Conv2D,
        predefined blocks. Train on MNIST, Evaluate population. Train/Test Split
        of 30/70 seems reasonable. Save best performing genes to Gene Record.
        This is one population cycle.

        Consider a multi-armed bandit controller for Gene Selection.

'''

'''

    Components are modular, individually immutable and smallest units.
    Dropout and batchnorm over all layers taken as standard procedure for each
    component.

'''


class Assembler:
    '''
        The Assembler Class
    '''
    def __init__(self, component_specs):
        self.component_specs = component_specs

    def assemble(self, gene):
        model_array = []
        for layer_set in gene["layers"]:
            layer_set_array = []
            for component_idx in gene["layers"][layer_set]['components']:
                # print(component_idx)
                component = self.component_specs["COMPONENT_Mapping"][component_idx]
                # print(component)
                # print(Model(component))
                layer_set_array.append(Model(component))
            model_array.append(layer_set_array)
        print(model_array)
