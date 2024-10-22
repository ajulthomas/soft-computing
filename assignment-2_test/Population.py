from random import randint, choice, choices, sample
from typing import List, Tuple

from Individual import Individual, Chromosome


class Population:
    def __init__(self, population_size, chromosome_length, generation, fitness_fn):
        self.population = []
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.best_individual: Individual = None
        self.generation = generation
        self.parents = []
        self.fitness_fn = fitness_fn

    def generate_individuals(self) -> Individual:
        chromosome: Chromosome = [1] * self.chromosome_length

        # total value of the chromosome
        target_sum = 80

        # remaining value to be randomly added
        remaining_sum = target_sum - sum(chromosome)

        # Randomly distribute the remaining sum among the numbers
        while remaining_sum > 0:
            index = randint(1, self.chromosome_length - 1)
            chromosome[index] += 1
            remaining_sum -= 1

        return Individual(chromosome=chromosome)

    def evaluate_population(self):
        self.best_individual = self.population[0]
        for individual in self.population:
            individual.calculate_fitness(self.fitness_fn)

        # sort the population based on fitness
        self.population = sorted(self.population, key=lambda x: x.fitness)

        for individual in self.population:
            # print(f"individual fitness: {individual.fitness}")
            if individual.fitness < self.best_individual.fitness:
                self.best_individual = individual

        print(f"Population best individual: {self.best_individual}")

    def initialize_population(self):
        self.population = [
            self.generate_individuals() for _ in range(self.population_size)
        ]
        self.evaluate_population()

    # function to print the generation, best individual and the fitness
    def __str__(self):
        return f"\nGeneration: {self.generation}\nBest Individual: {self.best_individual.chromosome},\nFitness: {self.best_individual.fitness}"
