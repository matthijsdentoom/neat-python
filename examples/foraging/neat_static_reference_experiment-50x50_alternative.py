"""
Simple example using the tile structure creation task.
"""

import os

from gym_multi_robot.envs.foraging_game_alternative import ClosestGame, WeightedSumForagingEnv

import neat
import gym

from examples.experiment_functions import NEATSwarmExperimentRunner
from examples.experiment_template import SingleExperiment

# Important variables.
experiment_name = 'NEAT_foraging_50x50_a'
num_steps = 3000
num_robots = 5
num_generations = 100
num_runs = 1
config_name = 'config-feedforward-50x50'

gym.register(
    id='foraging50x50-distance-sum-v0',
    entry_point='gym_multi_robot.envs:ForagingEnv',
    kwargs={'env_storage_path': 'foraging50x50.pickle', 'game_cls': ClosestGame}
)

if __name__ == '__main__':

    env = gym.make('foraging50x50-distance-sum-v0')
    runner = NEATSwarmExperimentRunner(env, num_steps)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = SingleExperiment(config, runner, num_generations, experiment_name, 2)

    for i in range(num_runs):
        experiment.run(experiment_name + str(num_runs))
