import numpy as np
import math

def individual(icls):
    genome = list()
    genome.append(np.random.uniform(-10, 10))
    genome.append(np.random.uniform(-10, 10))
    return icls(genome)


def ackleyFunc(chromosome):
    firstSum = 0.0
    secondSum = 0.0
    for c in chromosome:
        firstSum += c ** 2.0
        secondSum += math.cos(2.0 * math.pi * c)
        n = float(len(chromosome))
    return -20.0 * math.exp(-0.2 * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + 20 + math.e,