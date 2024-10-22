import service_simulator
import numpy as np
from typing import List
import numpy as np

# define employee setting as a list of integers
EmployeeSetting = List[int]

employee_setting_sample = [
    1,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    2,
    1,
    2,
    4,
    1,
    1,
    2,
    3,
    3,
    1,
    1,
    2,
    1,
    1,
    2,
    2,
    1,
    1,
    3,
    1,
    1,
    2,
    1,
    3,
    1,
    1,
    3,
    1,
    1,
    2,
    1,
    1,
    4,
    2,
    2,
    1,
    1,
    1,
]


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


def results(employee_setting=employee_setting_sample):
    _x, wait_times_per_service = service_simulator.init_and_simulate(employee_setting)
    sums = [
        np.sum(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    counts = [
        len(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    print("Mean waiting time for all services", np.sum(sums) / np.sum(counts))

    maximums = [
        np.max(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    print("Max time waited by a customer", np.max(maximums))

    print("fitness score", fitness_function(employee_setting))


for i in range(10):
    print(f"\nRun {i+1}")
    results()
    print("\n------------------------------------\n")
