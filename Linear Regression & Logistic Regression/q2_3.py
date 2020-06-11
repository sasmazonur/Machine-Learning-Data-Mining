#!/usr/bin/env python3
# q2_3.py
# Adrian Henle; CS434 Sp'20 Group 47
# Adapted from q2_1.py, written by Tristan
# Implements batch gradient descent for training a binary classifier via logistic regression
#  with L2 regularization.

import csv, sys, math
import numpy as np
import matplotlib.pyplot as plt


# Specified learning rate
learnRate = np.longdouble(1e-1)

# Number of training iterations
kmax = 10
	
# Sigmoid function
g = lambda x : 1/(1 + np.exp(-x))


# CLI args and hack-y defaults
try:
	trainData = sys.argv[1]
except:
	trainData = "usps_train.csv"
try:
	testData = sys.argv[2]
except:
	testData = "usps_test.csv"
try:
	lambdas = sys.argv[3]
except:
	lambdas = "lambdas.txt"
	
	
def normalize(array, m, n):
	for k in range (0, m):
		for i in range(0, n):
			array[k, i] /= n
	
	
def batch_gradient_logbin_l2(trainArray, testArray, n, train_m, test_m, trainY, testY, lambda_i):
	
	# Extract X and Y for test and train
	X_train = trainArray[:, :-1]
	Y_train = np.atleast_2d(trainArray[:, -1]).T
	# X_test = testArray[:, :-1] # Could be used for plotting, but isn't
	Y_test = testArray[:, -1]
	
	weightArray = np.atleast_2d(np.zeros(n))
	
	for k in range(0, kmax):
		gradient = np.atleast_2d(np.zeros(n))
		
		# Calculate gradient
		x = np.matmul(X_train, weightArray.T)
		a = np.matmul((g(x) - Y_train).T, X_train)
		gradient = (a + lambda_i*weightArray)/train_m
		
		# Update weights
		weightArray = weightArray - (learnRate * gradient)
		
		#calculate accuracy vs. training data
		#iterating through all 1400 examples in the training set
		successes = 0
		for i in range(0, train_m):
		
			dprod = 0
			#calculating dot product
			for m in range(0, n-1):
				dprod = dprod + (trainArray[i, m] * weightArray.T[m])
	
			#yhat = 1 / (1 + (math.e ** -dprod))
			yhat = g(dprod)
	
			#classification threshold for yes vs. no
			if yhat < 0.5:
				yhat = 0
			else:
				yhat = 1
	
			if yhat == Y_train[i]:
				successes = successes + 1
			
		#determine success ratio
		trainY[k] = successes/train_m
				
		#calculate accuracy vs. testing data
		#iterating through all 800 examples in the testing set
		successes = 0
		for i in range(0, test_m):
		
			dprod = 0
			#calculating dot product
			for m in range(0, n-1):
				dprod = dprod + (testArray[i, m] * weightArray.T[m])
	
			yhat = g(dprod)
	
			#classification threshold for yes vs. no
			if yhat < 0.5:
				yhat = 0
			else:
				yhat = 1
	
			if yhat == Y_test[i]:
				successes = successes + 1
		
		#determine success ratio
		testY[k] = successes/test_m
		
		
def plot_results(trainX, trainY, testX, testY):
	
	fig, ax = plt.subplots()
	ax.plot(trainX, trainY, label = "Training")
	ax.plot(testX, testY, label = "Test")
	ax.legend()
	plt.xlabel("Training Iterations")
	plt.ylabel("Accuracy")
	plt.title("Model Validation")
	
	plt.show()


def main():
	
	# Load data
	trainArray = np.genfromtxt(trainData, delimiter=',', dtype=np.longdouble)
	testArray = np.genfromtxt(testData, delimiter=',', dtype=np.longdouble)
	lambdaArray = np.genfromtxt(lambdas, delimiter=',', dtype=np.longdouble)
	
	# Data sizes
	train_m = np.size(trainArray, 0) # training examples
	test_m  = np.size(testArray,  0) # test cases
	n = np.size(trainArray, 1)-1 # number of features
	
	# Vectors for plotting/calculations
	trainX = [x for x in range(kmax)]
	trainY = np.zeros(kmax)
	testX  = [x for x in range(kmax)]
	testY  = np.zeros(kmax)	
	
	# Normalize features
	normalize(trainArray, train_m, n)
	normalize(testArray, test_m, n)
	
	for lambda_i in lambdaArray:
	
		print(lambda_i)
		
		# Gradient Descent Loop
		batch_gradient_logbin_l2(trainArray, testArray, n, train_m, test_m, trainY, testY, lambda_i)
	
		# Plot results
		plot_results(trainX, trainY, testX, testY)


if __name__ == "__main__":
	main()
