import random
import pandas
import time
from random_forest import trainTestSplit, createRandomForest, randomForestPredictions, calculateAccuracy

dataFrame = pandas.read_csv("dataset_files/breast_cancer.csv")
dataFrame = dataFrame.drop("id", axis=1)
dataFrame = dataFrame[dataFrame.columns.tolist()[1:] +
                      dataFrame.columns.tolist()[0: 1]]
dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize=0.25)

print("Random Forest - Breast Cancer Dataset")
print("  Maximum bootstrap size (n) is {}".format(dataFrameTrain.shape[0]))
print("  Maximum random subspace size (d) is {}".format(
    dataFrameTrain.shape[1] - 1))
