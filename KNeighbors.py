from sklearn import metrics
import random
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def rand_val(index):
    if index == 0:
        return random.randint(1, 10)
    elif index == 1:
        return ["uniform","distance"][random.randint(0, 1)]
    elif index == 2:
        return ["auto","ball_tree", "kd_tree", "brute"][random.randint(0, 3)]
    elif index == 3:
        return random.randint(10, 50)
    elif index == 4:
        return random.randint(1, 10)

def KNeighborsParametersFeatures(numberFeatures,icls):
    genome = list()
    for i in range(5):
        genome.append(rand_val(i))

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def KNeighborsParametersFeatureFitness(y, df, numberOfAttributes, individual):
    split=5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAttributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAttributes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures) 

    estimator = KNeighborsClassifier(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2], leaf_size=individual[3], p=individual[4])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel() 
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum += result
    return (resultSum / split,)

def mutationKNeighbors(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer <= 4:
        individual[numberParamer] = rand_val(numberParamer)
    else: #genetyzcna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0