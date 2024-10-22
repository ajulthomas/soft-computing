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
    population_size = 10
    chromosome_length = 50
    ga = GA.GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        fitness_fn=fitness_function,
        crossover_method="uniform",
    )
    ga.run()


if __name__ == "__main__":
    # main()
    service_simulator.init_and_simulate(80)


# class Evolution:
#     def __init__(self):
#         self.generations = []
#         self.current_generation = 0
#         self.largest_number = 100

#     def __str__(self):
#         return f"\nCurrent generation: {self.current_generation}, Largest number: {self.largest_number}\n"

#     def evolve(self):

#         # evolve the population
#         while self.current_generation < 10:
#             population = Population()
#             population.create_numbers()
#             self.generations.append(population)

#             # if the largest number in the population is greater than the current largest number
#             if population.largest_number < self.largest_number:
#                 self.largest_number = population.largest_number

#             self.current_generation += 1

#         print(self)


# class Population:
#     def __init__(self):
#         self.largest_number = 100
#         self.numbers = []

#     def create_numbers(self):
#         # randomly generate 10 numbers between 0 and 100
#         numbers = np.random.randint(0, 100, 10)
#         self.largest_number = min(numbers)
#         print(f"\nNumbers: {numbers}, Largest number: {self.largest_number}\n")


# Evolution().evolve()
