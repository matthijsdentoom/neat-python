"""
Simple example using the tile structure creation task.
"""

import os

import neat
import gym

from examples.experiment_functions import SMSwarmExperimentRunner
from examples.experiment_template import SingleExperiment

# Important variables.
from neat.state_machine_genome import StateMachineGenome

experiment_name = 'SM_4_states'
num_steps = 3000
num_robots = 5
num_generations = 100
num_runs = 5
num_trails = 5
config_name = 'config-state_machine'

if __name__ == '__main__':

    env = gym.make('tiling-pattern-v0')
    runner = SMSwarmExperimentRunner(env, num_steps)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(StateMachineGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = SingleExperiment(config, runner, num_generations, experiment_name, num_trails)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
