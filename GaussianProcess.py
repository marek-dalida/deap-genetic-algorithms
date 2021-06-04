from sklearn import metrics
import random
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier

def rand_val(index):
    if index == 0:
        return random.randint(1, 10)
    elif index == 1:
        return random.randint(10, 50)
    elif index == 2:
        return random.randint(0, 1) == 1
    elif index == 3:
        return random.randint(0, 1) == 1

def GaussianProcessParametersFeatures(numberFeatures,icls):
    genome = list()
    for i in range(4):
        genome.append(rand_val(i))

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def GaussianProcessParametersFeatureFitness(y, df, numberOfAttributes, individual):
    split=5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAttributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAttributes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures) 

    estimator = GaussianProcessClassifier(n_restarts_optimizer=individual[0], max_iter_predict=individual[1], warm_start=individual[2], copy_X_train=individual[3])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel() 
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum += result
    return (resultSum / split,)

def mutationGaussianProcess(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer <= 3:
        individual[numberParamer] = rand_val(numberParamer)
    else: #genetyzcna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0