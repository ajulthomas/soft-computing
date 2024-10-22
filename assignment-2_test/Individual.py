from random import randint, choice, choices, sample
from typing import List, Tuple


# chromosome - how a typical solution looks like
Chromosome = List[int]


class Individual:
    def __init__(self, chromosome: Chromosome):
        self.chromosome: Chromosome = chromosome
        self.fitness = 1000000

    def __str__(self):
        # print chromosome and fitness
        return f"Chromosome: {self.chromosome}\nFitness: {self.fitness}"

    def calculate_fitness(self, fitness_fn):
        self.fitness = fitness_fn(self.chromosome)

    # defines swap mutation
    def mutate(self, mutation_rate: float) -> Chromosome:
        if randint(0, 100) < mutation_rate * 100:
            l = len(self.chromosome)
            mutation_point_1 = randint(0, l - 1)
            mutation_point_2 = randint(0, l - 1)
            self.chromosome[mutation_point_1], self.chromosome[mutation_point_2] = (
                self.chromosome[mutation_point_2],
                self.chromosome[mutation_point_1],
            )
