import torch
from random import randint


# Chromosome type now uses CUDA tensors when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Individual:
    def __init__(self, chromosome: torch.Tensor):
        # Ensure the chromosome tensor is on the GPU if available
        self.chromosome = chromosome.to(device)
        self.fitness = torch.tensor(
            1e6, device=device
        )  # Fitness initialized on the GPU

    def __str__(self):
        # Print chromosome and fitness (converting tensors to lists and scalars)
        return f"Chromosome: {self.chromosome.tolist()}\nFitness: {self.fitness.item()}"

    def calculate_fitness(self, fitness_fn):
        # Fitness function now operates on CUDA tensors
        self.fitness = fitness_fn(self.chromosome).to(device)

    # Swap mutation using CUDA tensors
    def mutate(self, mutation_rate: float) -> torch.Tensor:
        if randint(0, 100) < mutation_rate * 100:
            l = self.chromosome.size(0)
            mutation_point_1 = randint(0, l - 1)
            mutation_point_2 = randint(0, l - 1)
            # Perform mutation using GPU-accelerated tensors
            self.chromosome[mutation_point_1], self.chromosome[mutation_point_2] = (
                self.chromosome[mutation_point_2].clone(),
                self.chromosome[mutation_point_1].clone(),
            )
