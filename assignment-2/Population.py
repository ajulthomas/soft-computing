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
            print(f"individual fitness: {individual.fitness}")
            if individual.fitness < self.best_individual.fitness:
                self.best_individual = individual

    def initialize_population(self):
        self.population = [
            self.generate_individuals() for _ in range(self.population_size)
        ]
        self.evaluate_population()

    # function to print the generation, best individual and the fitness
    def __str__(self):
        return f"\nGeneration: {self.generation}\nBest Individual: {self.best_individual.chromosome},\nFitness: {self.best_individual.fitness}"


import service_simulator
from typing import List
import numpy as np

# define employee setting as a list of integers
EmployeeSetting = List[int]


# define the fitness function
def fitness_function(employee_setting: EmployeeSetting) -> float:
    # if the total number of employee is not 80, return a high fitness value
    if sum(employee_setting) != 80:
        return 1000000

    # simulate the service with the given employee setting
    # and calculate the average waiting time
    _, wait_times_per_service = service_simulator.init_and_simulate(employee_setting)
    sums = [
        np.sum(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    counts = [
        len(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    mean_wait_time = np.sum(sums) / np.sum(counts)

    # maximum waiting time
    maximums = [
        np.max(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    max_wait_time = np.max(maximums)

    # create a fitness value giving 80% weight to the average waiting time
    # and 20% weight to the maximum waiting time
    fitness = 0.8 * mean_wait_time + 0.2 * max_wait_time

    return fitness


population = Population(
    population_size=10, chromosome_length=50, generation=0, fitness_fn=None
)


individual = population.generate_individuals()

for i in range(10):
    print(individual)
    fitness = fitness_function(individual.chromosome)
    print(f"fitness value: {fitness}")
