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
