import os
import json
import gym

from experiment import EvoStrat_Experiment

ENV = 'CartPole-v1'
input_dim = gym.make(ENV).observation_space.shape[0]
output_dim = gym.make(ENV).action_space.n

experiment_params = {
    'env_name': ENV,
    'model_structure': {"input":{"layer_size_mapping": {"in_features": input_dim,
                                                        "out_features": 64},
                               "layer_type": "linear",
                               "activation": "relu"},
                        "output":{"layer_size_mapping": {"in_features": 64,
                                                        "out_features": output_dim},
                               "layer_type": "linear",
                               "activation": "nil"}},
   'GENERATIONS': 15,
   'MAX_STEPS': 1000,
   'num_episodes': 20,
   'population_size': 20,
   'population_size_decay': 1,
   'minimum_population_size': 100,
   'elite_proportion': 0.2,
   'step_size': 0.4,
   'step_size_decay': 0.95,
   'minimum_step_size': 0.1,
   'lr_mean': 0.1
}

experiment = EvoStrat_Experiment(experiment_params)
solution_score = experiment.run_experiment()

PARAMS_PATH = './solutions/' + ENV + '_' + solution_score + '/model_params.json'

with open(PARAMS_PATH, 'w') as outfile:
    json.dump(experiment_params, outfile)
