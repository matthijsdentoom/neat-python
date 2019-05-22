import numpy as np


class PerceptronNetwork:
    """ This class resembles a perceptron network, meaning a neural network without hidden layers."""

    def __init__(self, weight_matrix, bias_vector, aggregation_function, activation_function):
        self.biases = np.array(bias_vector)
        self.weights = np.array(weight_matrix)
        self.agg_func = aggregation_function
        self.act_func = activation_function

    def activate(self, inputs):
        # Check that the inputs and the weightmatrix are of the same length.
        assert len(inputs) == len(self.weights[0])

        # Perform neural network operations.
        np_inputs = np.array(inputs)
        combined_weights = np.multiply(np_inputs, self.weights)

        # Note that this sum fails in case of a single row of weight
        aggregate = [self.agg_func(weight_row) for weight_row in combined_weights]
        assert len(aggregate) == len(self.biases)

        aggregate += self.biases  # Add biases
        return [self.act_func(output) for output in aggregate]  # Apply activation function.
