'''
    1. Spawn a set of progenitor genes using Spawn.py.
    2. Using these progenitors, recombine, shuffle, mutate, to produce a
       population of Child genes using Populate.py.
    3. For each component of each child gene, pull the corresponding NN
       hyperparam/optimizer/NN-component from Model.py.
    4. Arrange and assemble all components in Assemble.py, using tools defined
       in Tools.py.
    5. Each child gene takes the form of a dictionary:
       {Gene, Structure, Training_Details, Evaluation_Performance}.
       Dictionaries stored in an array.
    6. For each child, train for a set amount of time, evaluate on a test set,
       update Training_Details and Evaluation_Performance.
    7. For each population, rank by Evaluation_Performance, store the Genes,
       Training_Details, and Evaluation_Performance of the best members in the
       GeneRecord in a running .CSV, using Record.py
'''

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

    Major Challenges:

        1. Layer input/output adaptors. How do we feed variable sized inputs
           to several different layer types simultaneously? How do we connect
           a conv2D component to an FCNN to a Conv1D to a Conv2D again?
        2. Input and output should be fixed.

'''
import numpy as np
import torch
# import torch.utils.data.DataLoader as DataLoader

from Spawn import Spawner
from Assemble import Assembler
from Components import Component_Specs

from synth_data_prep import *
BASE_PATH =  './data'

train_path = BASE_PATH + '/data.json'
test_path = BASE_PATH + '/test_data.json'
training_datasets = prep_synth_data(train_path)[0]
testing_dataset = prep_synth_test_data(test_path)

INPUT_DIM = training_datasets[0].shape[1]
OUTPUT_DIM = 5

training_datasets = torch.utils.data.TensorDataset(training_datasets[0], training_datasets[1].float())
testing_dataset = torch.utils.data.TensorDataset(testing_dataset[0], testing_dataset[1].float())

trainloader = torch.utils.data.DataLoader(training_datasets, batch_size=45)
testloader = torch.utils.data.DataLoader(testing_dataset, batch_size=45)

test_spawner = Spawner()
test_gene = test_spawner.spawn_single_progenitor()
test_components = Component_Specs()
test_assembler = Assembler(test_components.return_component_mappings())
assembled_components = test_assembler.assemble_components(test_gene, (INPUT_DIM,), (OUTPUT_DIM,))

# test_assembler.insert_size_adapters(assembled_components, (INPUT_DIM,), (OUTPUT_DIM,))

# print(assembled_components)
# test_components = Component_Specs()
# # print(test_components.LR_Mapping)
#
# test_assembler = Assembler(test_components.return_component_mappings())
#
# assembled_components = test_assembler.assemble_components(test_gene)
# test_assembler.insert_size_adapters(assembled_components, (INPUT_DIM,), (OUTPUT_DIM,))
