# -- coding: utf-8 --
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier
sns.set()



def load_args():
	""" Loads arguments from command line """

	parser = argparse.ArgumentParser(description='arguments')

	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--decision_tree_plot', default=1, type=int) ## Remove?
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--random_forest_b', default=1, type=int)
	parser.add_argument('--random_forest_d', default=1, type=int)
	parser.add_argument('--random_forest_f', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	parser.add_argument('--DEBUG', default='', type=str)

	args = parser.parse_args()

	return args



def county_info(args):
	""" Loads the data label dictionary and displays it """
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)



def decision_tree_testing(x_train, y_train, x_test, y_test):
	""" Tests the DecisionTreeClassifier """
	print('\n\nDecision Tree')
	clf = DecisionTreeClassifier(max_depth=20)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))



## Should move loop to another function, and put the plotting code into utils.py
def decision_tree_plot(x_train, y_train, x_test, y_test):
	""" Plots the results of a series of tests """
	#Setting up our plot arrays
	plotX = []
	trainY = []
	testY = []
	f1Y = []

	#loop through 1-25 depth trees while placing the values into the plot arrays
	for k in range(1, 26):
		plotX.append(k)
		clf = DecisionTreeClassifier(max_depth=k)
		clf.fit(x_train, y_train)
		preds_train = clf.predict(x_train)
		preds_test = clf.predict(x_test)
		train_accuracy = accuracy_score(preds_train, y_train)
		test_accuracy = accuracy_score(preds_test, y_test)
		preds = clf.predict(x_test)
		f1_score = f1(y_test, preds)
		trainY.append(train_accuracy)
		testY.append(test_accuracy)
		f1Y.append(f1_score)

	#plotting
	plt.plot(plotX, trainY)
	plt.xlabel('Tree depth')
	plt.ylabel('Accuracy')
	plt.title('Training Accuracy')
	plt.show()

	plt.plot(plotX, testY)
	plt.xlabel('Tree depth')
	plt.ylabel('Accuracy')
	plt.title('Testing Accuracy')
	plt.show()

	plt.plot(plotX, f1Y)
	plt.xlabel('Tree depth')
	plt.ylabel('F1 Score')
	plt.title('F1 Score vs. Tree Depth')
	plt.show()



def random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

	#(b)
	# For max depth = 7, max features = 11 and n trees ∈ [10,20,...,200],
	# plot the train and testing accuracy of the forest versus the number of trees
	# in the forest n. Please also plot the train and testing F1 scores
	# versus the number of trees.
def random_forest_testing_b(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	#Setting up our plot arrays
	plotX = []
	trainY = []
	testY = []
	f1Y = []
	f1Y_train = []


	print("Running Part2.b")
	# Increment the sequence with 10. Starting at 10 end 200:
	# for num_tree in range(10, 40, 10):
	for num_tree in range(10, 210, 10):
		print("num_tree is:" + str(num_tree))
		plotX.append(num_tree)
		rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=num_tree)
		rclf.fit(x_train, y_train)
		preds_train = rclf.predict(x_train)
		preds_test = rclf.predict(x_test)
		train_accuracy = accuracy_score(preds_train, y_train)
		test_accuracy = accuracy_score(preds_test, y_test)
		preds = rclf.predict(x_test)
		preds_train = rclf.predict(x_train)

		f1_score = f1(y_test, preds)
		f1_score_train = f1(y_train, preds_train)
		trainY.append(train_accuracy)
		testY.append(test_accuracy)
		f1Y.append(f1_score)
		f1Y_train.append(f1_score_train)

	# plot the train and testing accuracy of the forest versus the number of trees
	# in the forest n.
	# plotting
	plt.plot(plotX, trainY, label='Training Data')
	plt.plot(plotX, testY, label='Testing Data')
	plt.xlabel('Number of Trees')
	plt.ylabel('Accuracy')
	plt.title('Accuracy versus the number of trees ')
	plt.legend()
	plt.show()

	plt.plot(plotX, f1Y, label='Testing Data')
	plt.plot(plotX, f1Y_train, label='Training Data')
	plt.xlabel('Number of Trees')
	plt.ylabel('F1 Score')
	plt.title('F1 Score vs. Tree Depth')
	plt.legend()
	plt.show()


def random_forest_testing_d(x_train, y_train, x_test, y_test):
		# (d) Repeat above experiments for max depth = 7, n = 50 trees,
		# and max features ∈ [1, 2, 5, 8, 10, 20, 25, 35, 50].
	 	# How does max features change the train/validation accuracy? Why?
		print("Running Part2.d")
		#Setting up our plot arrays
		plotX = []
		trainY = []
		testY = []
		f1Y = []
		f1Y_train = []
		for maxFeatures in [1, 2, 5, 8, 10, 20, 25, 35, 50]:
		# for maxFeatures in [1, 2, 5]:
			print("maxFeatures is:" + str(maxFeatures))
			plotX.append(maxFeatures)
			rclf = RandomForestClassifier(max_depth=7, max_features=maxFeatures, n_trees=50)
			rclf.fit(x_train, y_train)
			preds_train = rclf.predict(x_train)
			preds_test = rclf.predict(x_test)
			train_accuracy = accuracy_score(preds_train, y_train)
			test_accuracy = accuracy_score(preds_test, y_test)
			preds = rclf.predict(x_test)
			preds_train = rclf.predict(x_train)

			f1_score = f1(y_test, preds)
			f1_score_train = f1(y_train, preds_train)
			trainY.append(train_accuracy)
			testY.append(test_accuracy)
			f1Y.append(f1_score)
			f1Y_train.append(f1_score_train)

		# plot the train and testing accuracy of the forest versus the number of trees
		# in the forest n.
		# plotting
		plt.plot(plotX, trainY, label='Training Data')
		plt.plot(plotX, testY, label='Testing Data')
		plt.xlabel('Number of maxFeatures')
		plt.ylabel('Accuracy')
		plt.title('Accuracy versus the maxFeatures ')
		plt.legend()
		plt.show()

		plt.plot(plotX, f1Y_train, label='Training Data')
		plt.plot(plotX, f1Y, label='Testing Data')
		plt.xlabel("Number of maxFeatures")
		plt.ylabel('F1 Score')
		plt.title('F1 Score vs. maxFeatures')
		plt.legend()
		plt.show()

def random_forest_testing_f(x_train, y_train, x_test, y_test):
		# With your best result, run 10 trials with different random seeds (for the data/feature
		# sampling you can use the same seed) and report the individual train/testing accuracy’s
		# and F1 scores, as well as the average train/testing accuracy and F1 scores across the
		# 10 trials. Overall, how do you think randomness affects the performance?
		print("Running Part2.f")
		#Setting up our plot arrays
		plotX = []
		trainY = []
		testY = []
		f1Y = []
		f1Y_train = []
		for seed in range(10):
		# for maxFeatures in [1, 2, 5]:
			print("seed is:" + str(seed))
			plotX.append(seed)
			rclf = RandomForestClassifier(max_depth=4, max_features=9, n_trees=122)
			rclf.fit(x_train, y_train)
			preds_train = rclf.predict(x_train)
			preds_test = rclf.predict(x_test)
			train_accuracy = accuracy_score(preds_train, y_train)
			test_accuracy = accuracy_score(preds_test, y_test)
			preds = rclf.predict(x_test)
			preds_train = rclf.predict(x_train)

			f1_score = f1(y_test, preds)

			f1_score_train = f1(y_train, preds_train)
			trainY.append(train_accuracy)
			testY.append(test_accuracy)

			f1Y.append(f1_score)
			f1Y_train.append(f1_score_train)

		# Average
		#Dirty Printing. Put in table
		f1Y.append(sum(f1Y)/len(f1Y))
		f1Y_train.append(sum(f1Y_train)/len(f1Y_train))
		trainY.append(sum(trainY)/len(trainY))
		print('F1 {}'.format(f1Y))
		print('Test {}'.format(trainY))
		print('trainY {}'.format(trainY))




def ada_boost_testing(x_train, y_train, x_test, y_test):
	""" Tests AdaBoostClassifier """

	print("\n\nAdaBoost")

	# Number of weak learners
	L = [10*(i+1) for i in range(20)]
	R_tr = np.ones(0)
	R_te = np.ones(0)
	R_f1 = np.ones(0)
	for l in L:
		# Instantiate the classifier
		abc = AdaBoostClassifier(l)

		
		# Clean the data
		y_train[y_train == 0] = -1
		y_test[y_test == 0] = -1
		
		# Train the model
		abc.fit(x_train, y_train)
		

		# Test the model
		preds_train = abc.predict(x_train)
		#print(preds_train)
		#print(len(preds_train))
		#print(sum(preds_train == 1))
		preds_test = abc.predict(x_test)


		# Find the accuracy
		train_accuracy = accuracy_score(preds_train, y_train)
		test_accuracy = accuracy_score(preds_test, y_test)


		print("L = ", l)
		print('Train {}'.format(train_accuracy))
		print('Test {}'.format(test_accuracy))
		print('F1 Test {}'.format(f1(y_test, preds_test)))
		R_tr = np.append(R_tr,train_accuracy)
		R_te = np.append(R_te,test_accuracy)
		R_f1 = np.append(R_f1,f1(y_test, preds_test))

	print(L)
	plt.plot(L, R_tr)
	plt.plot(L, R_te)
	plt.plot(L, R_f1)
	plt.legend(["Train", "Test", "F1"])
	plt.title("AdaBoost")
	plt.xlabel("L")
	plt.ylabel("Score")
	plt.show()
	#print((y_test == preds_test).sum()/len(y_test))
	#print(sum(y_test == 1))
	#print(sum(preds_test == 1))



###############################################################################
#								MAIN BLOCK
###############################################################################
# TODO
## Assignment part 2 (b, c, d, [e], f)
## Assignment part 3 (e, f)
if __name__ == '__main__':

	# Get command line arguments
	args = load_args()

	# Handle the DEBUG arg
	DEBUG = args.DEBUG
	if DEBUG != '':
		for x in range(5):
			print("WARNING: DEBUG = {}\a".format(DEBUG))

	# Load data
	x_train, y_train, x_test, y_test = load_data(args.root_dir)

	# Run requested routines
	if (DEBUG == "ALL") | args.county_dict == 1:
		county_info(args)

	if (DEBUG == "ALL") | args.decision_tree == 1:
		decision_tree_testing(x_train, y_train, x_test, y_test)

	if (DEBUG == "ALL") | args.decision_tree_plot == 1:
		## Refactor so that each classifier_testing() always plots results?
		decision_tree_plot(x_train, y_train, x_test, y_test)

	if (DEBUG == "ALL") | args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)

	if (DEBUG == "ALL") | args.random_forest_b == 1:
			random_forest_testing_b(x_train, y_train, x_test, y_test)

	if (DEBUG == "ALL") | args.random_forest_d == 1:
			random_forest_testing_d(x_train, y_train, x_test, y_test)

	if (DEBUG == "ALL") | args.random_forest_f == 1:
			random_forest_testing_f(x_train, y_train, x_test, y_test)

	if (DEBUG == "ALL") | args.ada_boost == 1:
		ada_boost_testing(x_train, y_train, x_test, y_test)

	print('\n\nDone!\n\n\a')
