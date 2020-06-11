import pandas as pd
import numpy as np
import math
import csv
import sys
from lineaRegression import LineaRegression

#command line argument variables
trainData = sys.argv[1]
testData = sys.argv[2]

#Add the Labels
house_train = pd.read_csv(trainData, names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )
house_test = pd.read_csv(testData, names =['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV'] )

# Selecting first 13 columns of data frame with all rows
x_train = house_train.iloc[:,0:-1].values
x_test = house_test.iloc[:,0:-1].values

# Adding dummy variables
x_train = np.insert(x_train, 0, 1, axis=1)
x_test = np.insert(x_test, 0, 1, axis=1)

# Selecting last column of data frame for train/test data as matrix
y_train = (np.matrix(house_train.iloc[:,-1].values, dtype=float)).T
y_test = (np.matrix(house_test.iloc[:,-1].values, dtype=float)).T

#Do the Calculation
output = LineaRegression(np.matrix(x_train), y_train)
output2 = LineaRegression(np.matrix(x_test), y_test)

print("The learned weight vector:")
output.weight_vector()
print("\nASE over the training data")
trainAse = output.ase()
print(trainAse)
print("ASE over the Test data")
testAse = output2.ase()
print(testAse)
