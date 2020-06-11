import pandas as pd
import numpy as np
import math
import csv
import sys
import matplotlib.pyplot as plt
from lineaRegression import LineaRegression

#command line argument variables
trainData = sys.argv[1]
testData = sys.argv[2]

#Add the Labels
house_train = pd.read_csv(trainData, names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )
house_test = pd.read_csv(testData, names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )

#setting up python lists for plotting
trainX = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
testX = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
trainY = []
testY = []
d = 2
while d <= 20:

    #insert our 2 random sampled rows
    s1 = np.random.normal(0, 0.1, len(house_train.index))
    s2 = np.random.normal(0, 0.1, len(house_train.index))
    house_train.insert(0, "", s1, True)
    house_train.insert(0, "", s2, True)
    s1 = np.random.normal(0, 0.1, len(house_test.index))
    s2 = np.random.normal(0, 0.1, len(house_test.index))
    house_test.insert(0, "", s1, True)
    house_test.insert(0, "", s2, True)

    # Selecting all but last column of data frame with all rows
    x_train = house_train.iloc[:,0:-1].values
    x_test = house_test.iloc[:,0:-1].values

    # Selecting last column of data frame for train/test data as matrix
    y_train = (np.matrix(house_train.iloc[:,-1].values, dtype=float)).T
    y_test = (np.matrix(house_test.iloc[:,-1].values, dtype=float)).T

    #Do the Calculation
    output = LineaRegression(np.matrix(x_train), y_train)
    output2 = LineaRegression(np.matrix(x_test), y_test)

    trainY.append(output.ase())
    testY.append(output2.ase())

    d = d + 2

#plot graphs
plt.plot(trainX, trainY)
plt.xlabel('d')
plt.ylabel('ASE')
plt.title('Training ASE over different d')
plt.show()

plt.plot(testX, testY)
plt.xlabel('d')
plt.ylabel('ASE')
plt.title('Testing ASE over different D')
plt.show()
