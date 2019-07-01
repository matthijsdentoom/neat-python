from neat.config import DefaultClassConfig, ConfigParameter


class DefaultGeneration:
    """ This class represents generation by initializing n random individuals. """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict, [])

    def __init__(self, config, genome_indexer):
        self.config = config
        self.genome_indexer = genome_indexer

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g

        return new_genomes
