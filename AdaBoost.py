from sklearn import metrics
import random
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

def rand_val(index):
    if index == 0:
        return random.randint(10, 15)
    elif index == 1:
        return random.uniform(0.1, 2.5)
    elif index == 2:
        return ["SAMME","SAMME.R"][random.randint(0, 1)]

def AdaBoostParametersFeatures(numberFeatures,icls):
    genome = list()
    for i in range(3):
        genome.append(rand_val(i))

    for i in range(0,numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)

def AdaBoostParametersFeatureFitness(y, df, numberOfAttributes, individual):
    split=5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = []
    for i in range(numberOfAttributes, len(individual)):
        if individual[i] == 0:
            listColumnsToDrop.append(i - numberOfAttributes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures) 

    estimator = AdaBoostClassifier(n_estimators=individual[0], learning_rate=individual[1], algorithm=individual[2])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel() 
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum += result
    return (resultSum / split,)

def mutationAdaBoost(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer <= 2:
        individual[numberParamer] = rand_val(numberParamer)
    else: #genetyzcna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0