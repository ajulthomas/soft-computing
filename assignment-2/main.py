import genetic
from problem import fitness_function


# set the parameters for the genetic algorithm as a dictionary
parameters = {
    "chromosome_length": 50,
    "population_size": 100,
    "generations": 100,
    "crossover_probability": 0.8,
    "mutation_probability": 0.1,
    "tournament_size": 5,
}


# run the genetic algorithm
