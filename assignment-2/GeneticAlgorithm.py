from random import randint, choice, choices, sample
from typing import List, Tuple
import service_simulator
import numpy as np

from Individual import Individual, Chromosome
from Selection import Selection
from Crossover import Crossover
from Population import Population
import copy


class GeneticAlgorithm:
    def __init__(
        self,
        population_size,
        chromosome_length,
        fitness_fn,
        mutation_rate=0.01,
        patience=10,
        max_generations=100,
        selection_method="roulette",
        crossover_method="single",
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_fn = fitness_fn
        self.mutation_rate = mutation_rate
        self.generation_count = 0
        self.generations: List[Individual] = list()
        self.best_individual = None
        self.patience = patience
        self.max_patience = patience
        self.max_generations = max_generations
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.initialize()

    def initialize(self):
        print("initializing the population..........")
        population = Population(
            population_size=self.population_size,
            chromosome_length=self.chromosome_length,
            fitness_fn=self.fitness_fn,
            generation=self.generation_count,
        )
        population.initialize_population()

        # append the initial population to the generations
        self.generations.append(population)

        # set the best individual
        self.best_individual = self.generations[self.generation_count].best_individual

    def __str__(self):
        return f"\nGA Summary\nGeneration: {self.generation_count},\nBest Individual: {self.best_individual.chromosome},\nFitness: {self.best_individual.fitness}"

    def select_parents(self) -> List:

        # get the current generation
        g = self.generation_count

        # get the previous population
        population = self.generations[g - 1].population

        # get the other parents size
        parents_size = (self.population_size - 2) // 2
        parents = []

        if self.selection_method == "tournament":
            parents = [
                Selection.tournament_selection(population, 3)
                for _ in range(parents_size)
            ]

        elif self.selection_method == "roulette":
            parents = [
                Selection.roulette_wheel_selection(population)
                for _ in range(parents_size)
            ]

        # print(f"selected parentpopulations size: {len(parents)}")
        self.generations[g].parents = parents

    def crossover(self):

        print("initiating crossover..........")

        # get the current generation
        g = self.generation_count

        # get the parents
        parents = self.generations[g].parents

        # get the elites from the previous generation
        elites = self.generations[g - 1].population[:2]

        # get the crossover method
        crossover_method = self.crossover_method

        # add elites to the children
        children = elites

        if crossover_method == "single":
            for parent1, parent2 in parents:
                child1, child2 = Crossover.single_point_crossover(parent1, parent2)
                children.extend([child1, child2])

        elif crossover_method == "two":
            for parent1, parent2 in parents:
                child1, child2 = Crossover.two_point_crossover(parent1, parent2)
                children.extend([child1, child2])

        elif crossover_method == "uniform":
            for parent1, parent2 in parents:
                child1, child2 = Crossover.uniform_crossover(parent1, parent2)
                children.extend([child1, child2])

        # length of the children
        # print(f"children size: {len(children)}")
        self.generations[g].population = children

    def mutate(self):
        # get the current generation
        g = self.generation_count

        for individual in self.generations[g].population:
            individual.mutate(self.mutation_rate)

    def evaluate_generation(self):
        # get current population
        current_population: Population = self.generations[self.generation_count]

        # evaluate the population
        current_population.evaluate_population()

        # population best individual
        cp_best_individual = copy.deepcopy(current_population.best_individual)

        # generation best individual
        gn_best_individual = self.best_individual

        if cp_best_individual.fitness < gn_best_individual.fitness:

            # changes the best individual
            self.best_individual = cp_best_individual

            # resets the patience
            self.patience = self.max_patience
        else:
            self.patience -= 1

    def evolve(self):
        # create new population instrance
        current_population = Population(
            population_size=self.population_size,
            chromosome_length=self.chromosome_length,
            generation=self.generation_count,
            fitness_fn=self.fitness_fn,
        )
        self.generations.append(current_population)

        self.select_parents()
        self.crossover()
        self.mutate()
        self.evaluate_generation()

    def run(self):
        while self.generation_count < self.max_generations:
            self.generation_count += 1
            self.evolve()

            # if self.generation_count == 20:
            #     break
        print(self)


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


def main():
    population_size = 10
    chromosome_length = 50
    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        fitness_fn=fitness_function,
        crossover_method="uniform",
    )
    ga.run()


if __name__ == "__main__":
    main()
