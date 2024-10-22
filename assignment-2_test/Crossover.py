from random import randint, choice, choices, sample
from typing import List, Tuple
from Individual import Individual


class Crossover:
    """Class to handle crossover in genetic algorithms."""

    @staticmethod
    def single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple:
        """Performs single point crossover on two parents."""
        crossover_point = randint(1, len(parent1.chromosome) - 1)

        child1 = []
        child2 = []

        # while sum(child1) != 80:
        child1 = (
            parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        )

        # while sum(child2) != 80:
        child2 = (
            parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        )

        return Individual(child1), Individual(child2)

    @staticmethod
    def two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple:
        """Performs two point crossover on two parents."""
        crossover_points = sorted(sample(range(1, len(parent1.chromosome) - 1), 2))

        child1 = []
        child2 = []

        # while sum(child1) != 80:
        child1 = (
            parent1.chromosome[: crossover_points[0]]
            + parent2.chromosome[crossover_points[0] : crossover_points[1]]
            + parent1.chromosome[crossover_points[1] :]
        )

        # while sum(child2) != 80:
        child2 = (
            parent2.chromosome[: crossover_points[0]]
            + parent1.chromosome[crossover_points[0] : crossover_points[1]]
            + parent2.chromosome[crossover_points[1] :]
        )

        return Individual(child1), Individual(child2)

    @staticmethod
    def uniform_crossover(parent1: Individual, parent2: Individual) -> Tuple:
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
