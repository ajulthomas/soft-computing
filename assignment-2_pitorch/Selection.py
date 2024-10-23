import torch
from random import sample
from typing import List, Tuple


class Selection:

    @staticmethod
    def tournament_selection(population: List, tournament_size: int) -> Tuple:
        """Selects two parents using tournament selection (fitness maximization)."""
        # Sample tournament_size individuals from the population and select the best
        selected = sample(population, tournament_size)
        parent1 = max(
            selected, key=lambda ind: ind.fitness.item()
        )  # .item() to extract scalar from tensor

        # Repeat for second parent
        selected = sample(population, tournament_size)
        parent2 = max(selected, key=lambda ind: ind.fitness.item())

        return parent1, parent2

    @staticmethod
    def roulette_wheel_selection(population: List) -> Tuple:
        # Convert fitness values to tensors and use PyTorch to handle the sum
        fitness_values = torch.tensor(
            [ind.fitness.item() for ind in population],
            device=population[0].fitness.device,
        )

        # Compute the total fitness to derive selection probabilities
        total_fitness = torch.sum(fitness_values)
        selection_probs = (
            fitness_values / total_fitness
        )  # Probability for each individual

        # Convert probabilities back to CPU to use Python choices function
        selection_probs_cpu = selection_probs.cpu().numpy()

        # Select two parents using weighted selection based on fitness
        parent1 = torch.multinomial(
            torch.tensor(selection_probs_cpu), num_samples=1, replacement=True
        ).item()
        parent2 = torch.multinomial(
            torch.tensor(selection_probs_cpu), num_samples=1, replacement=True
        ).item()

        return population[parent1], population[parent2]
