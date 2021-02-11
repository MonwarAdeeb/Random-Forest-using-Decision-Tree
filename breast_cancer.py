import random
import pandas
import time
from random_forest import trainTestSplit, createRandomForest, randomForestPredictions, calculateAccuracy

dataFrame = pandas.read_csv("dataset_files/breast_cancer.csv")
dataFrame = dataFrame.drop("id", axis=1)
dataFrame = dataFrame[dataFrame.columns.tolist()[1:] +
                      dataFrame.columns.tolist()[0: 1]]
dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize=0.25)
