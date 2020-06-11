import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not used here?
from sklearn.metrics import f1_score
# import seaborn as sns # Not used here?
# sns.set()


# Pulls test and train data sets from the indicated directory
def load_data(rootdir='./'):
	x_train = np.loadtxt(rootdir+'x_train.csv', delimiter=',').astype(int)
	y_train = np.loadtxt(rootdir+'y_train.csv', delimiter=',').astype(int)
	x_test = np.loadtxt(rootdir+'x_test.csv', delimiter=',').astype(int)
	y_test = np.loadtxt(rootdir+'y_test.csv', delimiter=',').astype(int)
	y_train[y_train == -1] = 0
	y_test[y_test == -1] = 0
	return x_train, y_train, x_test, y_test


# Pulls the data label dictionary from the indicated directory
def load_dictionary(rootdir='./'):
	county_dict = pd.read_csv(rootdir+'county_facts_dictionary.csv')
	return county_dict


# Prints the info contained in the data label dictionary
def dictionary_info(county_dict):
	for i in range(county_dict.shape[0]):
		print('Feature: {} - Description: {}'.format(i, county_dict['description'].iloc[i]))


# Returns accuracy of predicted y-values
def accuracy_score(preds, y):
	#print(len(preds))
	accuracy = np.array(preds == y).sum()/len(y)
	return accuracy


# Returns the f1 score using sklearn.metrics
def f1(y, yhat):
	return f1_score(y, yhat)


def weighted_pred(y, W):
	#print(W is None)
	total = sum([y[i]*W[i] for i in range(len(W))])
	if total < 0:
		return -1
	else:
		return 1


###########################################################################
# TODO
# Plotting/processing functions, etc
###########################################################################


