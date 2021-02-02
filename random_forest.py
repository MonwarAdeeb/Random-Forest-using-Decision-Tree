import numpy
import pandas
import random
from decisionTree import buildDecisionTree, decisionTreePredictions


def trainTestSplit(dataFrame, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize * len(dataFrame))
    indices = dataFrame.index.tolist()
    testIndices = random.sample(population=indices, k=testSize)
    dataFrameTest = dataFrame.loc[testIndices]
    dataFrameTrain = dataFrame.drop(testIndices)
    return dataFrameTrain, dataFrameTest


def bootstrapSample(dataFrame, bootstrapSize):
    randomIndices = numpy.random.randint(
        low=0, high=len(dataFrame), size=bootstrapSize)
    return dataFrame.iloc[randomIndices]
