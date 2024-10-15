from random import randint, choice
from typing import List


# the possible values for genes
GENES = [i for i in range(1, 32)]

# chromosome - how a typical solution looks like
Chromosome = List[int]

# population - a collection of chromosomes
Population = List[Chromosome]

# fitness - how good a solution is


# function to generate the chromosome / solution
# it takes length of the genome sequence as input
def generate_chromosome(l: int) -> Chromosome:
    chromosome: Chromosome = [1] * l

    # total value of the chromosome
    target_sum = 80

    # remaining value to be randomly added
    remaining_sum = target_sum - sum(chromosome)

    # Randomly distribute the remaining sum among the numbers
    while remaining_sum > 0:
        index = randint(1, l - 1)
        chromosome[index] += 1
        remaining_sum -= 1

    return chromosome


# function to gnerate the initial population
def generate_initial_population(
    chromosome_size: int, population_size: int
) -> Population:
    return [generate_chromosome(chromosome_size) for _ in range(population_size)]


# mutation - change a random gene in the chromosome
# implments swap mutation
def mutate(chromosome: Chromosome, mutation_rate: float) -> Chromosome:
    # if the mutation rate is less than the random value, return the chromosome
    if randint(0, 100) < mutation_rate * 100:
        # get the length of the chromosome
        l = len(chromosome)

        # get the mutation point
        mutation_point_1 = randint(0, l - 1)

        # get the mutation point
        mutation_point_2 = randint(0, l - 1)

        # swap the two genes
        chromosome[mutation_point_1], chromosome[mutation_point_2] = (
            chromosome[mutation_point_2],
            chromosome[mutation_point_1],
        )

    return chromosome


# single point crossover - combine two chromosomes
def single_point_crossover(
    chromosome1: Chromosome, chromosome2: Chromosome
) -> Chromosome:
    # get the length of the chromosome
    l = len(chromosome1)

    # get the crossover point
    crossover_point = randint(1, l - 1)

    # combine the two chromosomes
    return chromosome1[:crossover_point] + chromosome2[crossover_point:]


# multi-point crossover - combine two chromosomes at multiple points
def multi_point_crossover(
    chromosome1: Chromosome, chromosome2: Chromosome
) -> Chromosome:
    # get the length of the chromosome
    l = len(chromosome1)

    # get the crossover points
    crossover_points = [randint(1, l - 1) for _ in range(3)]

    # sort the crossover points
    crossover_points.sort()

    # combine the two chromosomes
    return (
        chromosome1[: crossover_points[0]]
        + chromosome2[crossover_points[0] : crossover_points[1]]
        + chromosome1[crossover_points[1] : crossover_points[2]]
        + chromosome2[crossover_points[2] :]
    )


# generate a generic crossover function
def crossover(
    chromosome1: Chromosome, chromosome2: Chromosome, type: str
) -> Chromosome:
    crossover_func = {
        "single": single_point_crossover,
        "multi": multi_point_crossover,
    }

    child_chromosome: Chromosome = []
    crossover_attempts = 0

    while sum(child_chromosome) != 80:
        child_chromosome = crossover_func[type](chromosome1, chromosome2)
        crossover_attempts += 1

    return child_chromosome, crossover_attempts


# tournament selection
def tournament_selection(population: Population, tournament_size: int) -> Chromosome:
    # get the length of the population
    l = len(population)

    # get the tournament participants
    participants = [choice(population) for _ in range(tournament_size)]

    # get the best participant
    best_participant = min(participants, key=lambda x: fitness(x))

    return best_participant


# evolve the population
def evolve(
    population: Population, mutation_rate: float, crossover_type: str
) -> Population:
    new_population: Population = []

    # sort the population based on the fitness
    population = sorted(population, key=lambda x: fitness(x))

    # keep the top 10% of the population
    new_population.extend(population[: int(0.1 * len(population))])

    # generate the children
    while len(new_population) < len(population):
        # select two parents
        parent1 = choice(population)
        parent2 = choice(population)

        # crossover
        child, crossover_attempts = crossover(parent1, parent2, crossover_type)

        # mutation
        if randint(0, 100) < mutation_rate * 100:
            child = mutate(child)

        # add the child to the new population
        new_population.append(child)

    return new_population
