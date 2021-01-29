import numpy
import pandas
import random


def trainTestSplit(dataFrame, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize * len(dataFrame))
    indices = dataFrame.index.tolist()
    testIndices = random.sample(population=indices, k=testSize)
    dataFrameTest = dataFrame.loc[testIndices]
    dataFrameTrain = dataFrame.drop(testIndices)
    return dataFrameTrain, dataFrameTest


def checkPurity(data):
    if len(numpy.unique(data[:, -1])) == 1:
        return True
    else:
        return False


def classifyData(data):
    uniqueClasses, uniqueClassesCounts = numpy.unique(
        data[:, -1], return_counts=True)
    return uniqueClasses[uniqueClassesCounts.argmax()]


def getPotentialSplits(data, randomAttributes):
    potentialSplits = {}
    _, columns = data.shape
    columnsIndices = list(range(columns - 1))
    if randomAttributes != None and len(randomAttributes) <= len(columnsIndices):
        columnsIndices = randomAttributes
    for column in columnsIndices:
        values = data[:, column]
        uniqueValues = numpy.unique(values)
        if len(uniqueValues) == 1:
            potentialSplits[column] = uniqueValues
        else:
            potentialSplits[column] = []
            for i in range(len(uniqueValues)):
                if i != 0:
                    currentValue = uniqueValues[i]
                    previousValue = uniqueValues[i - 1]
                    potentialSplits[column].append(
                        (currentValue + previousValue) / 2)
    return potentialSplits


def splitData(data, splitColumn, splitValue):
    splitColumnValues = data[:, splitColumn]
    return data[splitColumnValues <= splitValue], data[splitColumnValues > splitValue]


def calculateEntropy(data):
    _, uniqueClassesCounts = numpy.unique(data[:, -1], return_counts=True)
    probabilities = uniqueClassesCounts / uniqueClassesCounts.sum()
    return sum(probabilities * -numpy.log2(probabilities))


def calculateOverallEntropy(dataBelow, dataAbove):
    pDataBelow = len(dataBelow) / (len(dataBelow) + len(dataAbove))
    pDataAbove = len(dataAbove) / (len(dataBelow) + len(dataAbove))
    return pDataBelow * calculateEntropy(dataBelow) + pDataAbove * calculateEntropy(dataAbove)


def determineBestSplit(data, potentialSplits, randomSplits=None):
    overallEntropy = 9999
    bestSplitColumn = 0
    bestSplitValue = 0
    if randomSplits == None:
        for splitColumn in potentialSplits:
            for splitValue in potentialSplits[splitColumn]:
                dataBelow, dataAbove = splitData(data, splitColumn, splitValue)
                currentOverallEntropy = calculateOverallEntropy(
                    dataBelow, dataAbove)
                if currentOverallEntropy <= overallEntropy:
                    overallEntropy = currentOverallEntropy
                    bestSplitColumn = splitColumn
                    bestSplitValue = splitValue
    else:
        for i in range(randomSplits):
            randomSplitColumn = random.choice(list(potentialSplits))
            randomSplitValue = random.choice(
                potentialSplits[randomSplitColumn])
            dataBelow, dataAbove = splitData(
                data, randomSplitColumn, randomSplitValue)
            currentOverallEntropy = calculateOverallEntropy(
                dataBelow, dataAbove)
            if currentOverallEntropy <= overallEntropy:
                overallEntropy = currentOverallEntropy
                bestSplitColumn = randomSplitColumn
                bestSplitValue = randomSplitValue
    return bestSplitColumn, bestSplitValue
