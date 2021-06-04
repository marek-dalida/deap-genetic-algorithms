import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
import random
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def SVCParametersFeatures(numberFeatures,icls):
    genome = list()
    # kernel
    listKernel = ["linear","rbf", "poly", "sigmoid"]
    genome.append(listKernel[random.randint(0, 3)])
    #c
    k = random.uniform(0.1, 100)
    genome.append(k)
    #degree
    genome.append(random.uniform(0.1,5))
    #gamma
    gamma = random.uniform(0.001,5)
    genome.append(gamma)
    # coeff
    coeff = random.uniform(0.01, 10)
    genome.append(coeff)
    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def SVCParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split=5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAttributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAttributes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures) 

    estimator = SVC(kernel=individual[0],C=individual[1],degree=individual[2],gamma=individual[3],coef0=individual[4],random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
        predicted).ravel() 
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum += result
    return (resultSum / split,)

def mutationSVC(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer==0:
        # kernel
        listKernel = ["linear", "rbf", "poly", "sigmoid"]
        individual[0]=listKernel[random.randint(0, 3)]
    elif numberParamer==1:
        #C
        k = random.uniform(0.1,100)
        individual[1]=k
    elif numberParamer == 2:
        #degree
        individual[2]=random.uniform(0.1, 5)
    elif numberParamer == 3:
        #gamma
        gamma = random.uniform(0.01, 5)
        individual[3]=gamma
    elif numberParamer ==4:
        # coeff
        coeff = random.uniform(0.1, 20)
        individual[2] = coeff
    else: #genetyzcna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0

pd.set_option('display.max_columns', None)
df = pd.read_csv('data.csv', sep=',')

y = df['Status']
df.drop('Status', axis=1, inplace=True)

df.drop('ID', axis=1, inplace=True)
df.drop('Recording', axis=1, inplace=True)

numberOfAttributes = len(df.columns)

mms = MinMaxScaler()
df_norm = mms.fit_transform(df)

clf = SVC()
scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
print(scores.mean())

mean_population_cost = []
std_population_cost = []

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", SVCParametersFeatures, numberOfAttributes, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", SVCParametersFeatureFitness, y, df, numberOfAttributes)
toolbox.register("select", tools.selBest)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutationSVC)

sizePopulation = 50
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration = 20

start_timestamp = datetime.now()
pop = toolbox.population(n=sizePopulation)
fitnesses = list(map(toolbox.evaluate, pop))

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
