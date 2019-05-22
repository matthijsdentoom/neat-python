from random import random

import numpy as np
from neat.state_machine_genome import StateMachineGenome

from neat.genes import BaseGene
from neat.state_machine_genes import StateGene

from neat.state_machine_attributes import BiasesAttribute, WeightsAttribute

from neat.six_util import iteritems
from neat.state_machine_base_genome import StateMachineBaseGenomeConfig


class SelectorStateGene(BaseGene):
    """
    This state gene represents the gene of a state in a state selector network, so it has a seperate nn,
    which selects the states.
    """

    _non_added_attributes = [BiasesAttribute('selector_biases', 'selector_bias'),
                             WeightsAttribute('selector_weights', 'selector_weight')]

    def __init__(self, key, num_states):
        assert isinstance(key, int), "StateGene key must be an int, not {!r}".format(key)
        super(SelectorStateGene, self).__init__(key)
        self.num_states = num_states
        self.biases = None
        self.weights = None

    @classmethod
    def get_config_params(cls):
        params = BaseGene.get_config_params()
        params.extend(attr.get_config_params() for attr in cls._non_added_attributes)

    def init_attributes(self, config):
        self.biases = self._non_added_attributes[0].init_value(config, self.num_states)
        self.weights = self._non_added_attributes[1].init_value(config, self.num_states)

    def mutate(self, config):
        self.biases = self._non_added_attributes[0].mutate_value(self.biases, config)
        self.weights = self._non_added_attributes[1].mutate_value(self.weights, config)

    def copy(self):
        state = SelectorStateGene(self.key, self.num_states)
        state.biases = np.array(self.biases)
        state.weights = np.array(self.weights)

        return state


class StateSelectorGenome:
    """ Genome of the state selector, which selects the state to go to based on a seperate nn."""

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = StateGene
        param_dict['connection_gene_type'] = SelectorStateGene
        return StateMachineBaseGenomeConfig(param_dict)

    def __init__(self, key):
        self.key = key
        self.states = {}
        self.selectors = {}
        self.aggregation = 'sum'
        self.activation = 'sigmoid'
        self.fitness = None

    def configure_new(self, config):
        """" Create a simple state machine without any outgoing states. """
        for i in range(config.num_initial_states):
            self.states[i] = StateMachineGenome.create_state(config, i)

        # Create the selector states
        for i in range(config.num_initial_states):
            self.selectors[i] = self.create_selector(config, i, config.num_initial_states)

    @staticmethod
    def create_selector(config, state_key, num_states):

        state = SelectorStateGene(state_key, num_states)
        state.init_attributes(config)
        return state

    def clone(self, genome):
        """ This function clones the given genome in the current genome. """
        for key, state in iteritems(genome.states):
            self.states[key] = state.copy()

        for key, selector in iteritems(genome.selectors):
            self.selectors[key] = selector.copy()

    def configure_crossover(self, genome1, genome2, config):
        """ So far no good crossover has been identified for this task."""
        pass

    def mutate(self, config):

        if len(self.states) < config.max_num_states and random() < config.state_add_prob:
            self.mutate_add_state(config)

        if random() < config.state_delete_prob:
            self.mutate_delete_state(config)

        # Mutate node genes (bias, response, etc.).
        for state in self.states.values():
            state.mutate(config)

        for selector in self.selectors.values():
            selector.mutate(config)

    def mutate_add_state(self, config):

        state_key = config.get_new_node_key()
        new_state = StateMachineGenome.create_state(config, state_key)
        new_selector = self.create_selector(config, state_key, len(self.states) + 1)

        # Enter values in dictionary.
        self.states[state_key] = new_state
        self.selectors[state_key] = new_selector

        # TODO: also add state to selector networks of other states.

    def mutate_delete_state(self, config):
        pass
        # TODO: also remove from other states.

    def size(self):
        return len(self.states)

    def distance(self, other, _):
        return abs(len(self.states) - len(other.states))

    def __str__(self):

        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.states):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\n"
        return s