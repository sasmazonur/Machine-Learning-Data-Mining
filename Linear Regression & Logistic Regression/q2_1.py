import csv
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

#command line argument variables
trainData = sys.argv[1]
testData = sys.argv[2]
learnRate = np.longdouble(sys.argv[3])

#load up the training and testing data into two different arrays
trainArray = np.genfromtxt(trainData, delimiter=',', dtype=np.longdouble)
testArray = np.genfromtxt(testData, delimiter=',', dtype=np.longdouble)

#initialize several vectors for plotting/calculations
#keep in mind numpy arrays are still 0-based indexing
weightArray = np.zeros(256)
trainX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
trainY = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
testX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
testY = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#normalizing all features by dividing by 255
for k in range (0, 1399):
    for i in range(0, 255):
        trainArray[k, i] = trainArray[k, i] / 255

#normalizing all features by dividing by 255
for k in range (0, 799):
    for i in range(0, 255):
        testArray[k, i] = testArray[k, i] / 255

#our batch learning loop
#epochs
for k in range(0, 10):
    gradient = np.zeros(256)

    #iterating through all 1400 examples in the training set
    for i in range(0, 1399):

        dprod = 0
        #calculating dot product
        for m in range(0, 255):
            dprod = dprod + (trainArray[i, m] * weightArray[m])

        yhat = 1 / (1 + (math.e ** -dprod))

        #calculating gradient vector
        for j in range(0, 255):
            gradient[j] = gradient[j] + ((yhat - trainArray[i, 256]) * trainArray[i, j])

    #update weights
    for l in range(0, 255):
        weightArray[l] = weightArray[l] - (learnRate * gradient[l])

    #calculate accuracy vs. training data
    #iterating through all 1400 examples in the training set
    successes = 0
    for i in range(0, 1399):
    
        dprod = 0
        #calculating dot product
        for m in range(0, 255):
            dprod = dprod + (trainArray[i, m] * weightArray[m])

        yhat = 1 / (1 + (math.e ** -dprod))

        #classification threshold for yes vs. no
        if yhat < 0.5:
            yhat = 0
        else:
            yhat = 1

        if yhat == trainArray[i, 256]:
            successes = successes + 1
        
    #determine success ratio
    trainY[k] = successes/1400
            
    #calculate accuracy vs. testing data
    #iterating through all 800 examples in the testing set
    successes = 0
    for i in range(0, 799):
    
        dprod = 0
        #calculating dot product
        for m in range(0, 255):
            dprod = dprod + (testArray[i, m] * weightArray[m])

        yhat = 1 / (1 + (math.e ** -dprod))

        #classification threshold for yes vs. no
        if yhat < 0.5:
            yhat = 0
        else:
            yhat = 1

        if yhat == testArray[i, 256]:
            successes = successes + 1
    
    #determine success ratio
    testY[k] = successes/800

plt.plot(trainX, trainY)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training Data Accuracy')
plt.show()

plt.plot(testX, testY)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Testing Data Accuracy')
plt.show()




