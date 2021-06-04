from sklearn import metrics
import random
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def rand_val(index):
    if index == 0:
        return ["gini","entropy"][random.randint(0, 1)]
    elif index == 1:
        return random.randint(1, 50)
    elif index == 2:
        return random.randint(1, 10)
    elif index == 3:
        return random.randint(1, 10)

def RandomForestParametersFeatures(numberFeatures,icls):
    genome = list()
    genome.append(rand_val(0))
    genome.append(rand_val(1))
    genome.append(rand_val(2))
    genome.append(rand_val(3))

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def RandomForestParametersFeatureFitness(y, df, numberOfAttributes, individual):
    split=5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAttributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAttributes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures) 

    estimator = RandomForestClassifier(criterion=individual[0], n_estimators=individual[1], max_depth=individual[2], min_samples_leaf=individual[3])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel() 
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum += result
    return (resultSum / split,)

def mutationRandomForest(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        individual[0] = rand_val(0)
    elif numberParamer==1:
        individual[1] = rand_val(1)
    elif numberParamer == 2:
        individual[2] = rand_val(2)
    elif numberParamer == 3:
        individual[3] = rand_val(3)
    else: #genetyzcna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0