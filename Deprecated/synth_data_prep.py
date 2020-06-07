import json
import torch as th
import random

def prep_synth_data(path):
    """
    This function loads a prepared dataset organized into a dictionary format,
    shuffles it, and organizes it into the format expected by simulated FL.

    Args:
        path (string): Path to json file containing data in dictionary format.

    Returns:
        training_datasets (Dict): Keys are client indices, values are tuples
                                  where first entry is X, second is y.
    """
    training_datasets = {}
    with open(path) as json_file:
        data = json.load(json_file)
        for client_idx in data:
            num_data = len(data[client_idx]['x'])
            x_and_y = list(zip(data[client_idx]['x'],
                               data[client_idx]['y']))
            random.shuffle(x_and_y)
            x, y = zip(*x_and_y)
            # training_datasets.update({int(client_idx) : (th.tensor(x),
            #                                              th.tensor(y).view(-1, 1))})
            training_datasets.update({int(client_idx) : (th.tensor(x),
                                                         th.tensor(y))})
    return training_datasets

def prep_synth_test_data(path):
    """
    This function is identical to previous, without the indexing by client.

    Args:
        path (string): Path to json file containing data in dictionary format.

    Returns:
        testing_dataset (tuple): First entry is X, second is y.
    """
    testing_dataset = None
    with open(path) as json_file:
        data = json.load(json_file)
        x_and_y = list(zip(data['x'],
                           data['y']))
        random.shuffle(x_and_y)
        x, y = zip(*x_and_y)
        # return (th.tensor(x), th.tensor(y).view(-1, 1))
        return (th.tensor(x), th.tensor(y))
