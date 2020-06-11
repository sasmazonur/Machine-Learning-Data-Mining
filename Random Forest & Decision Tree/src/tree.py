import numpy as np
import random
import math

from utils import weighted_pred

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree



class DecisionTreeClassifier():
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	# take in features X and labels y
	# build a tree

	def fit(self, X, y, W = None, featureIDX = None):
		self.num_classes = len(set(y))
		if W is not None:
				self.root = self.build_tree(X, y, 1, W, featureIDX = featureIDX)
		else:
			self.root = self.build_tree(X, y, 1,featureIDX = featureIDX)

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree

	def build_tree(self, X, y, depth, W = None, featureIDX = None):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		if featureIDX == None:
			self.features_idx = np.arange(0, X.shape[1])
		else:
			self.features_idx = featureIDX

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		if W is None:
			num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
			prediction = np.argmax(num_samples_per_class)
		else:
			prediction = weighted_pred(y, W)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y

		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:

			########################################
			# Assignment part 1 (b)
			########################################
			# count all the values we need for our calculations
			clm, clp = [np.sum(left_y == i) for i in range(self.num_classes)]
			crm, crp = [np.sum(right_y == i) for i in range(self.num_classes)]
			ym, yp = [np.sum(y == i) for i in range(self.num_classes)]

			# find our uncertainty measures
			uy = 1 - (yp / (ym + yp))**2 - (ym / (yp + ym))**2
			ul = 1 - (clp / (clp + clm))**2 - (clm / (clm + clp))**2
			ur = 1 - (crp / (crp + crm))**2 - (crm / (crm + crp))**2

			# find the split probabilities
			pl = (clp + clm) / (yp + ym)
			pr = (crp + crm) / (yp + ym)

			# benefit of split calculation
			gain = uy - (pl * ul) - (pr * ur)

			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0



class RandomForestClassifier():
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth
		self.ensemble = []

	# fit all trees
	def fit(self, X, y):
		bagged_X, bagged_y = self.bag_data(X, y)
		featureIDX = []
		# print('Fitting Random Forest...\n')
		# print("self.n_trees is:")
		# print(self.n_trees)
		# print("self.max_features is:")
		# print(self.max_features)
		# X contains 51 categorical features
		catagoircalFeatureSize = 51
		for i in range(self.n_trees):
			featureIDX = random.sample(range(catagoircalFeatureSize), self.max_features)
			treeBo = DecisionTreeClassifier(max_depth = self.max_depth)
			treeBo.fit(bagged_X[i], bagged_y[i], featureIDX = featureIDX)
			self.ensemble.append(treeBo)

	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		# data set of size 2098
		y_size = len(X)

		# Pick some sample of rows and features with replacement
		# new_data < data
		# Decision Tree 1.Low Bias, 2.High Variance
		# Random Forest - when we combine all it will lead to Low Variance
		for i in range(self.n_trees):
			# It returns an array of specified shape and fills it with random integers from 0-2097
			r1 = np.random.random_integers(0, high=y_size-1, size=y_size)
			bagged_X.append(X[r1])
			bagged_y.append(y[r1])

		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)


	def predict(self, X):
		preds = np.zeros(len(X)).astype(int)

		for i in range(self.n_trees):
			preds = preds + self.ensemble[i].predict(X)

		for i in range(len(preds)):
			if preds[i] >= self.n_trees/2:
				preds[i] = 1
			else:
				preds[i] = 0

		return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier():

	
	
	def __init__(self, T = 1):
		self.H = [DecisionTreeClassifier(1) for t in range(T)] # Weak learner array
		self.A = [0 for t in range(T)] # Learner weights
		
		
	def predict(self, X):
		F = np.ndarray(X.shape[0])
		P = np.ndarray((X.shape[0], len(self.H)))
		for j in range(len(self.H)):
			preds = self.H[j].predict(X)
			#print("Preds: ", len(preds))
			#print(sum(preds == 1))
			#print("H: ", len(self.H))
			#print("X: ", X.shape)
			for i in range(X.shape[0]):
				P[i][j] = self.A[j]*preds[i]
		for i in range(X.shape[0]):
			F[i] = np.sign(sum(P[i]))
		#print("F: ", len(F))
		return F
	
	
	def fit(self, X, Y):
		self.n = len(X)
		W = np.array([1/self.n for x in range(self.n)])
		
		for t in range(len(self.H)):
			# Choose strongest weak learner
			self.H[t].fit(X, Y, W) ## Change to use W on X
			
			# Determine learner's alpha
			self.A[t] = self._alpha(self._epsilon(self.H[t], X, Y, W))
			
			# Update weights
			W = self._update_weights(self.A[t], self.H[t], X, Y, W)
			#print("W: ",W)
	
	
	def _update_weights(self, alpha, h, X, Y, W):
		preds = h.predict(X)
		M = [W[i]*np.exp(-Y[i]*alpha*preds[i]) for i in range(self.n)]
		s = sum(M)
		return [M[i]/s for i in range(self.n)]
	
	
	def _epsilon(self, h, x, y, W):
		#x = np.reshape(x, (-1, np.size(x, 1)))
		#print(x.shape)
		#print(x[0].T.shape)
		preds = h.predict(x)
		idx_f = [i for i in range(self.n) if preds[i] != y[i]]
		return sum([W[i] for i in idx_f])
	
	
	def _alpha(self, e):
		return np.log((1-e)/e)/2

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			

