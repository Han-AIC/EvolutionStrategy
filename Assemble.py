
import numpy as np
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
    def __init__(self):
        pass

class MetaComponent:
    '''
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
    def __init__(self):
        pass

class ModelComponent:
    '''
        The ModelComponent Class uses the metacomponent to select an NN component
        for each section of each layer.

        Model Component: {
            1: FCNN_64x64_ReLu,
            2: Cnv1D_64_64_3,
            3: Cnv2D_64_64_3,
        } 

    '''
    def __init__(self):
        pass
