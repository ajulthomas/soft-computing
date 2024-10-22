import GeneticAlgorithm as GA
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


def main():
    # create a genetic algorithm object
    genetic_algorithm = GA.GeneticAlgorithm(
        population_size=100,
        chromosome_length=50,
        fitness_fn=fitness_function,
        crossover_method="two",
        selection_method="roulette",
        mutation_rate=0.1,
    )

    # run the genetic algorithm
    genetic_algorithm.run()


if __name__ == "__main__":
    main()
