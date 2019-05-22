from neat.state_selector_network import StateSelectorNetwork

from neat.nn.feed_forward import FeedForwardNetwork
from neat.state_machine_network import StateMachineNetwork


class ExperimentRunner:
    """ This class represents an experiment runner, so an instance which can be used to run gym experiments.
        Note that the fitness is only requested using the get_fitness() function of the environment after the last step.
        So make sure the gym environment provides this.
    """

    def __init__(self, gym_environment, num_steps, controller_class, render=False):
        self.env = gym_environment
        self.num_steps = num_steps
        self.controller_class = controller_class
        self.render = render
        self.controller = None

    def run_multiple_trails(self, genome, config, num_trails):
        """ This function runs multiple trials with the same genome and environment.
            This can be useful when the environments contains randomness, since multiple situations are evaluated.
            returns the winner and the stats for each of the trials.
        """
        reward = 0
        for _ in range(num_trails):
            reward += self.run(genome, config)

        return reward / num_trails

    def run(self, genome, config):
        """ This function should run the experiment with the given genome and configuration.
            Should be implemented by the subclasses with an implementation of how to run.
            Returned should be the fitness of swarm behaviour, as indicated by the environment.
        """
        self.controller = self.controller_class()
        self.controller.reset(genome, config)
        observation = self.env.reset()
        fitness = 0

        for i in range(self.num_steps):

            output = self.controller.step(observation)
            observation, fitness, done, _ = self.env.step(output)

            self.check_render()
            if done:
                break

        return fitness

    def draw(self, genome, config, file_name='winner.svg'):
        """ This function should draw the given genome. It depends on the genome that is actually used."""
        self.controller_class.draw(genome, config, file_name)

    def check_render(self):
        if self.render:
            self.env.render()


class SwarmExperimentRunner(ExperimentRunner):
    """ This class describes an experiment runner for a swarm, ie. multiple robots with the same controller."""

    def run(self, genome, config):

        observations = self.env.reset()

        # Spawn as many controllers as there are observations.
        self.controller = [self.controller_class() for _ in range(len(observations))]

        # Reset all controllers with the genome and config.
        for controller in self.controller:
            controller.reset(genome, config)

        for i in range(self.num_steps):

            output = [self.controller[i].step(observations[i]) for i in range(len(observations))]
            observations, _, done, _ = self.env.step(output)

            self.check_render()
            if done:
                break

        return self.env.get_fitness()


class SimulationController:
    """ This class calculates the actions in the simulation using the given control mechanism."""

    def __init__(self):
        self.net = None

    def reset(self, genome, config):
        """ This function resets the stepper indicating that a new simulation is started"""
        pass

    def step(self, observation):
        """ This function calculates the desired course of action based on the given observation."""
        pass

    def get_states(self):
        """ This function returns the states the robot has been in. (if not implemented it returns an empty list."""
        return []


class FeedForwardNetworkController(SimulationController):
    """ This class calculates the next actions based on a feed forward network."""

    def reset(self, genome, config):
        self.net = FeedForwardNetwork.create(genome, config)

    def step(self, observation):
        return self.net.activate(observation)


class StateMachineController(SimulationController):
    """ This class calculates the next actions based on a state machine. The difference here is that the current state,
        needs to be taken into account and updated, as this is also part of the state machine.
    """

    def __init__(self, controller_cls):
        SimulationController.__init__(self)
        self.controller_cls = controller_cls
        self.current_state = 0
        self.state_logger = []

    def reset(self, genome, config):
        self.net = self.controller_cls.create(genome, config.genome_config)
        self.current_state = 0
        self.state_logger = []

    def step(self, observation):
        new_state, actions = self.net.activate(self.current_state, observation)
        self.current_state = new_state
        self.state_logger.append(self.current_state)    # Append the current state to the logger.

        return actions

    def get_states(self):
        return self.state_logger


class SMControllerFactory:
    """ This class generates state machine controllers."""

    def __init__(self, controller_cls):
        self.controller_cls = controller_cls

    def generate(self, genome, config):
        controller = StateMachineController(self.controller_cls)
        controller.reset(genome, config)
        return controller

    def __str__(self):
        return 'Controller factory with {0}.'.format(self.controller_cls.__name__)


class FFControllerFactory:
    """ This class generates neural network controllers. """

    def generate(self, genome, config):
        controller = FeedForwardNetworkController()
        controller.reset(genome, config)
        return controller

    def __str__(self):
        return 'Controller factory with feed-forward controller.'
