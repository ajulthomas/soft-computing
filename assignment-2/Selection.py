from random import randint, choice, choices, sample
from typing import List, Tuple


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
