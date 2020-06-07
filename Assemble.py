from collections import defaultdict
import numpy as np

from Model import Model
import torch
import torch.autograd as autograd
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
def compute_out_size(in_size, mod):
    """
    Compute output size of Module `mod` given an input with size `in_size`.
    """

    f = mod.forward(autograd.Variable(torch.Tensor(1, *in_size)))
    return int(np.prod(f.size()[1:]))

class Assembler:
    '''
        The Assembler Class
    '''
    def __init__(self, component_specs):
        self.component_specs = component_specs

    def assemble_components(self, gene, input_dims, output_dims):
        print("-----------")
        print(gene)
        print(input_dims)
        print(output_dims)
        model_array = []
        size_array = [[input_dims]]
        for component_layer in gene["layers"]:
            component_layer_array = []
            component_output_size_array = []
            for component_tuple in gene["layers"][component_layer]['components']:
                component = self.component_specs["COMPONENT_Mapping"][component_tuple[0]]
                activation = self.component_specs["ACTIVATION_Mapping"][component_tuple[1]]
                component_name = list(component.keys())[0]
                component_output_size_array.append(component[component_name]["layer_size_mapping"]["out_features"])
                component[component_name].update({"in_features": activation})
                component[component_name].update({"activation": activation})
                
            size_array.append(component_output_size_array)
        print(size_array)
        # model_array = []
        # for component_layer in gene["layers"]:
        #     component_layer_array = []
        #     for component_tuple in gene["layers"][component_layer]['components']:
        #         component = self.component_specs["COMPONENT_Mapping"][component_tuple[0]]
        #         activation = self.component_specs["ACTIVATION_Mapping"][component_tuple[1]]
        #
        #
        #
        #
        #         component_name = list(component.keys())[0]
        #         component[component_name].update({"activation": activation})
        #         component_layer_array.append(Model(component))
        #     model_array.append(component_layer_array)
        # return model_array

    def insert_size_adapters(self, model_arr, input_dims, output_dims):
        print(model_arr)
        print(input_dims)
        print(output_dims)
        print("-------------")

        processed_model_arr = []
        for layer_components in model_arr:
            print(layer_components)
            # component_name = list(component.state_dict().keys())[0].split('.')[0]
            # component_shape = component.state_dict()[list(component.state_dict().keys())[0]].shape

            # print(component_name)
            # if component_name == "FCNN":
            #     print(component_shape)
            # elif component_name == "CONV1D":
            #     print(component_shape)
            # elif component_name == "CONV2D":
            #     print(component_shape)

        # for component_layer in model_arr:
        # #     print(component_layer)
        # #     print("-------------")
        #     for component in component
        #     print(component.state_dict().keys())
            # print(compute_out_size((64,64, 3), component))
        # for component in model_arr[-1]:
        #     print(component)


# class Individual(nn.Module):
#   def __init__(self,
#                model_arr):
#     super(Individual, self).__init__()
#     self.model_arr = model_arr
#
#   def forward(self, x):
#
#     for component_layer in self.model_arr:
#         intermediate_outputs = []
#         for component in component_layer:
#             intermediate_outputs.append(component(x))
#
#
#     x = self.prime_model(x)
#     value = self.value_model(x)
#     advantage = self.advantage_model(x)
#     Q = value + (advantage - advantage.mean())
#     return Q
