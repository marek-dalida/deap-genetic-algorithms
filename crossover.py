import numpy as np

def arithmetic_crossover(first, second):
    k = np.random.random()
    map_k = lambda a, b: k * a + (1-k) * b
    first_result = [map_k(first[0], second[0]), map_k(first[1], second[1])]
    second_result = [map_k(second[0], first[0]), map_k(second[1], first[1])]
    return first_result, second_result

def heuristic_crossover(first, second):
    k = np.random.random()
    map_k = lambda a, b: k * (b-a) + a
    result = [map_k(first[0], second[0]), map_k(first[1], second[1])]
    return result, None