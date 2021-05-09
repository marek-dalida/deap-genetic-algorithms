from deap import base
from deap import creator
from deap import tools
from utils import ackleyFunc, individual
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
from crossover import arithmetic_crossover, heuristic_crossover

import numpy as np

mean_population_cost = []
std_population_cost = []

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", ackleyFunc)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", arithmetic_crossover)
toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.2)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=16)
    toolbox.register("map", pool.map)

    sizePopulation = 100
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 100

    start_timestamp = datetime.now()
    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    numberElitism = 1
    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if np.random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if np.random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        mean_population_cost.append(mean)
        std_population_cost.append(std)

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    calc_time = datetime.now() - start_timestamp
    print("-- End of (successful) evolution --")

    print("\n Time of calculations: {}".format(calc_time))
    plt.plot(mean_population_cost)
    plt.xlabel("Epoch number")
    plt.ylabel("Mean population cost")
    plt.title("Mean population cost for each epoch")
    plt.savefig("results/mean_plot_{}.jpeg".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
    plt.show()
    plt.clf()

    plt.plot(std_population_cost)
    plt.xlabel("Epoch number")
    plt.ylabel("Standard deviation cost")
    plt.title("Standard deviation cost for each epoch")
    plt.savefig("results/std_plot_{}.jpeg".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
    plt.show()
    plt.clf()

    pool.close()
