"""
Allows the client to run a hyperparameter search with keras.
"""

import itertools
from logger.py import Logger

class HyperparameterSearch:
    def __init__(self, name):
        self.logger = Logger(name)

    """
    Runs the search. Tests a certain percentage of all possible permuattaions of the parameters.
    training_function(parameter) has to be a function which trains a model with keras and returns a history object.
    parameter has to be a dictionary with possible parameters.
    """
    def scan(self, training_function, parameter, percentage_tested):
        parameter_permutations = self.get_permutations(parameter, percentage_tested)
        number_rounds = len(parameter_permutations)

        for r in range(rounds):
            result = training_function(parameter_permutations[r])
            self.log_results(parameter_permutations[r], result)

    def get_permutations(self, parameter, percentage):
        keys, values = zip(*parameter.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        return permutations

    def log_results(self, parameter, result):
        self.logger.log(str(parameter) + " " + )