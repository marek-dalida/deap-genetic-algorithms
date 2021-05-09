import math
from constants import *

class AlgorithmConfig:
    def __init__(self, begin=-10, end=10, population_size=100, epoch_number=100, selection=SEL_TOURNAMENT,
                 crossover_func=CR_ARITHMETIC_CROSSOVER, mutation_func=MUT_GAUSSIAN, mutation_prop=0.2, crossover_prob=0.8,
                 elite_amount=0):
        self.begin = begin
        self.end = end
        self.population_size = population_size
        self.epoch_number = epoch_number
        self.selection = selection
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.mutation_prob = mutation_prop
        self.crossover_prob = crossover_prob
        self.elite_amount = elite_amount


def ackleyFunc(chromosome):
    firstSum = 0.0
    secondSum = 0.0
    for c in chromosome:
        firstSum += c ** 2.0
        secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(chromosome))
    return -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + 20 + math.e,


class ProblemDefinition:
    def __init__(self):
        self.func = ackleyFunc
