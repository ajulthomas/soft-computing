import service_simulator
import numpy as np
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# define employee setting as a list of integers
EmployeeSetting = List[int]


def results(
    employee_setting: EmployeeSetting, crossover_method: str, selection_method: str
) -> None:
    _x, wait_times_per_service = service_simulator.init_and_simulate(employee_setting)
    sums = [
        np.sum(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    counts = [
        len(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]

    print(
        f"\n GA Results [crossover_method: {crossover_method}, selection_method: {selection_method}] : \n"
    )

    print(
        f"-------------------------------------------------------------------------------------------------\n"
    )

    print("Best Employee Setting: ", employee_setting)

    print("Mean waiting time for all services", np.sum(sums) / np.sum(counts))

    maximums = [
        np.max(wait_times_per_service[j]) for j in range(len(wait_times_per_service))
    ]
    print("Max time waited by a customer", np.max(maximums))

    print(
        f"\n-------------------------------------------- END ------------------------------------------------\n"
    )


def plot_results(
    fitness_scores: List[float], crossover_method: str, selection_method: str
) -> None:
    # generations
    x = [i for i in range(1, len(fitness_scores) + 1)]

    # fitness scores
    y = fitness_scores

    plt.plot(x, y)
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.title(
        f"Fitness Score vs Generation [crossover_method: {crossover_method}, selection_method: {selection_method}]"
    )
    plt.show()

    # save the plot to './output/fitness_score_vs_generation_{crossover_method}_{selection_method}.png
    plt.savefig(
        f"./output/fitness_score_vs_generation_{crossover_method}_{selection_method}.png"
    )


# for i in range(10):
#     print(f"\nRun {i+1}")
#     results()
#     print("\n------------------------------------\n")


# employee_setting_sample = [
#     1,
#     1,
#     2,
#     1,
#     2,
#     1,
#     2,
#     1,
#     1,
#     2,
#     1,
#     1,
#     2,
#     1,
#     2,
#     4,
#     1,
#     1,
#     2,
#     3,
#     3,
#     1,
#     1,
#     2,
#     1,
#     1,
#     2,
#     2,
#     1,
#     1,
#     3,
#     1,
#     1,
#     2,
#     1,
#     3,
#     1,
#     1,
#     3,
#     1,
#     1,
#     2,
#     1,
#     1,
#     4,
#     2,
#     2,
#     1,
#     1,
#     1,
# ]
