import random
import pandas
import time
from randomForest import trainTestSplit, createRandomForest, randomForestPredictions, calculateAccuracy

dataFrame = pandas.read_csv("dataset_files/car_evaluation.csv")

buyingMapping = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
dataFrame["buying"] = dataFrame["buying"].map(buyingMapping)

maintMapping = {"low": 1, "med": 2, "high": 3, "vhigh": 4}
dataFrame["maint"] = dataFrame["maint"].map(maintMapping)

doorsMapping = {"2": 2, "3": 3, "4": 4, "5more": 5}
dataFrame["doors"] = dataFrame["doors"].map(doorsMapping)
