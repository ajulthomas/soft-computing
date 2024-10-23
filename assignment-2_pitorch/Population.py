import torch
from random import randint
from Individual import (
    Individual,
)  # Assuming this is the CUDA-based Individual class with Chromosome as a tensor

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # Initialize chromosome as a PyTorch tensor on the GPU
        chromosome = torch.ones(
            self.chromosome_length, dtype=torch.int32, device=device
        )

        # Total value of the chromosome
        target_sum = 80

        # Remaining value to be randomly added
        remaining_sum = target_sum - torch.sum(chromosome).item()

        # Randomly distribute the remaining sum among the numbers
        while remaining_sum > 0:
            index = randint(1, self.chromosome_length - 1)
            chromosome[index] += 1
            remaining_sum -= 1

        return Individual(chromosome=chromosome)

    def evaluate_population(self):
        # Evaluate fitness and sort individuals based on fitness
        self.best_individual = self.population[0]

        for individual in self.population:
            # Calculate fitness using the fitness function on the GPU
            individual.calculate_fitness(self.fitness_fn)

        # Sort population based on fitness
        self.population = sorted(self.population, key=lambda x: x.fitness.item())

        # Find the best individual
        for individual in self.population:
            if individual.fitness < self.best_individual.fitness:
                self.best_individual = individual

        print(f"Population best individual: {self.best_individual}")

    def initialize_population(self):
        # Generate initial population and evaluate them
        self.population = [
            self.generate_individuals() for _ in range(self.population_size)
        ]
        self.evaluate_population()

    def __str__(self):
        # Print generation, best individual, and fitness
        return f"\nGeneration: {self.generation}\nBest Individual: {self.best_individual.chromosome.cpu().tolist()},\nFitness: {self.best_individual.fitness.item()}"
