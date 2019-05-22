from neat.six_util import itervalues
from neat.state_machine_network import State


class StateSelectorNetwork:
    """ This class gives a neural network, which for every state has a perceptron network which selects the next state
    to go to."""

    def __init__(self, states, selectors):

        self.states = dict()
        self.selectors = dict()

        for state in states:
            if state.id in self.states:
                raise ValueError("State included twice")

            self.states[state.id] = state

        for selector in selectors:
            if selector.id in self.selectors:
                raise ValueError("State included twice")

            self.selectors[selector.id] = selector

        self.state_keys = list(self.states)     # Create a static list of all keys of the states in the network.

    def activate(self, current_state_id, inputs):

        # First check whether a state transition is required.
        next_state = self.eval_selector(current_state_id, inputs)

        # Evaluate the neural network of the current state.
        current_state = self.states[next_state]
        output = current_state.activate(inputs)

        return next_state, output

    def eval_selector(self, current_state_id, inputs):
        # Evaluate the neural network.
        state_scores = self.selectors[current_state_id].activate(inputs)
        assert len(state_scores) == len(self.state_keys)

        # Select the index with the highest score and return it.
        state_index = state_scores.index(max(state_scores))
        return self.state_keys[state_index]

    @staticmethod
    def create(genome, config):
        # This method creates a state selector network.
        assert len(genome.states) == len(genome.selectors)

        network_states = []
        for state_gene in itervalues(genome.states):
            network_states.append(State.create(state_gene, config))

        network_selectors = []
        for selector_gene in itervalues(genome.selectors):
            network_selectors.append(State.create(selector_gene, config))

        return StateSelectorNetwork(network_states, network_selectors)

