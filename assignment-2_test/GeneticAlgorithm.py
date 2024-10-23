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
        # print("initializing the population..........")
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
        parents_size = (self.population_size - 20) // 2
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

        # print("initiating crossover..........")

        # get the current generation
        g = self.generation_count

        # get the parents
        parents = self.generations[g].parents

        # get the elites from the previous generation
        elites = self.generations[g - 1].population[:20]

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
        gn_best_individual = copy.deepcopy(self.best_individual)

        if cp_best_individual.fitness < gn_best_individual.fitness:

            # print about the change
            print(
                f"\nChanging GA best individual from {gn_best_individual.fitness} to {cp_best_individual.fitness}\n"
            )

            # changes the best individual
            self.best_individual = copy.deepcopy(cp_best_individual)

            # reset the patience
            self.patience = self.max_patience

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

    def stop_condition(self, fitness_scores):
        # stop if the fitness score is not improving
        if len(fitness_scores) > 1:
            if fitness_scores[-1] == fitness_scores[-2]:
                self.patience -= 1

        # stop if patience is 0
        if self.patience == 0:
            return True

        return False

    def run(self):
        # variable to store the fitness score over the generations
        fitness_scores = []
        while self.generation_count < self.max_generations:
            self.generation_count += 1
            print(f"\nGeneration: {self.generation_count}")
            print("\n----------------------------------------\n")
            self.evolve()
            print(self)
            fitness_scores.append(self.best_individual.fitness)

            # stop if the fitness score is not improving
            if self.stop_condition(fitness_scores):
                break

        return fitness_scores, self.best_individual.chromosome
