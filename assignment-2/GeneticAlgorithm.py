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
        return f"Chromosome: {self.chromosome}, Fitness: {self.fitness}"

    def __repr__(self):
        return self.chromosome

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


class Population:
    def __init__(self, population_size, chromosome_length, generation, fitness_fn):
        self.population = []
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fittest = None
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

    def initialize_population(self):
        self.population = [
            self.generate_individuals() for _ in range(self.population_size)
        ]
        self.evaluate_population()

    def evaluate_population(self):
        self.fittest = self.population[0]
        for individual in self.population:
            individual.calculate_fitness(self.fitness_fn)
            if individual.fitness < self.fittest.fitness:
                self.fittest = individual

        # sort the population based on fitness
        self.population = sorted(self.population, key=lambda x: x.fitness)

    # function to print the generation, best individual and the fitness
    def __str__(self):
        return f"Generation: {self.generation}, Best Individual: {self.fittest.chromosome}, Fitness: {self.fittest.fitness}"


class Selection:

    @staticmethod
    def tournament_selection(population: List, tournament_size: int) -> Tuple:
        """Selects two parents using tournament selection."""
        selected = sample(population, tournament_size)
        parent1 = max(selected, key=lambda ind: ind.fitness)

        selected = sample(population, tournament_size)
        parent2 = max(selected, key=lambda ind: ind.fitness)

        return parent1, parent2

    @staticmethod
    def roulette_wheel_selection(population: List) -> Tuple:
        total_fitness = sum(ind.fitness for ind in population)
        selection_probs = [ind.fitness / total_fitness for ind in population]

        parent1 = choices(population, weights=selection_probs, k=1)[0]
        parent2 = choices(population, weights=selection_probs, k=1)[0]

        return parent1, parent2


class Crossover:
    """Class to handle crossover in genetic algorithms."""

    @staticmethod
    def single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple:
        """Performs single point crossover on two parents."""
        crossover_point = randint(1, len(parent1.chromosome) - 1)

        child1 = (
            parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        )
        child2 = (
            parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        )

        return Individual(child1), Individual(child2)

    @staticmethod
    def two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple:
        """Performs two point crossover on two parents."""
        crossover_points = sorted(sample(range(1, len(parent1.chromosome) - 1), 2))

        child1 = (
            parent1.chromosome[: crossover_points[0]]
            + parent2.chromosome[crossover_points[0] : crossover_points[1]]
            + parent1.chromosome[crossover_points[1] :]
        )
        child2 = (
            parent2.chromosome[: crossover_points[0]]
            + parent1.chromosome[crossover_points[0] : crossover_points[1]]
            + parent2.chromosome[crossover_points[1] :]
        )

        return Individual(child1), Individual(child2)

    @staticmethod
    def uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple:
        """Performs uniform crossover on two parents."""
        child1 = []
        child2 = []

        for gene1, gene2 in zip(parent1.chromosome, parent2.chromosome):
            if randint(0, 1):
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)

        return Individual(child1), Individual(child2)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size,
        chromosome_length,
        fitness_fn,
        mutation_rate=0.01,
        patience=10,
        max_generations=100,
        selection_method="tournament",
        crossover_method="single",
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_fn = fitness_fn
        self.mutation_rate = mutation_rate
        self.generation_count = 0
        self.generations = []
        self.best_individual = None
        self.patience = patience
        self.max_generations = max_generations
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.initialize()

    def initialize(self):
        population = Population(
            self.population_size, self.chromosome_length, self.generation_count,
            self.fitness_fn
        )
        population.initialize_population()
        # append the initial population to the generations
        self.generations.append(population)
        self.best_individual = self.generations[0].fittest
        self.generation_count += 1
    
    def __str__(self):
        return f"Generation: {self.generation_count}, Best Individual: {self.best_individual.chromosome}, Fitness: {self.best_individual.fitness}"

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
                for _ in range(parents_size // 2)
            ]

        elif self.selection_method == "roulette":
            parents = [
                Selection.roulette_wheel_selection(population)
                for _ in range(parents_size // 2)
            ]

        self.generations[g].parents = parents

    def crossover(self):

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

        self.generations[g].population = children

    def mutate(self):
        # get the current generation
        g = self.generation_count

        for individual in self.generations[g].population:
            individual.mutate(self.mutation_rate)

    def evaluate_generation(self):
        self.generations[self.generation_count].evaluate_population()

    def evolve(self):
        # initialise the generation
        self.generations.append(
            Population(
                self.population_size, self.chromosome_length, self.generation_count, self.fitness_fn
            )
        )
        self.select_parents()
        self.crossover()
        self.mutate()
        self.evaluate_generation()
        print(self.generations[self.generation_count])

    def run(self):

        patience_count = 0
        while self.generation_count < self.max_generations:
            self.evolve()

            # check if the best individual has not changed for patience generations
            if (
                self.best_individual.fitness
                == self.generations[self.generation_count].fittest.fitness
            ):
                patience_count += 1
            else:
                patience_count = 0
                # update the best individual
                if (
                    self.generations[self.generation_count].fittest.fitness
                    > self.best_individual.fitness
                ):
                    self.best_individual = self.generations[
                        self.generation_count
                    ].fittest

            if patience_count == self.patience:
                break
