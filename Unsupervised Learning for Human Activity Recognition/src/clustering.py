import numpy as np

class KMeans():
	"""
	KMeans. Class for building an unsupervised clustering model
	"""

	def __init__(self, k, max_iter=20):

		"""
		:param k: the number of clusters
		:param max_iter: maximum number of iterations
		"""

		self.k = k
		self.max_iter = max_iter

	def init_center(self, x):
		"""
		initializes the center of the clusters using the given input
		:param x: input of shape (n, m)
		:param k: the number of clusters
		:return: updates the self.centers
		"""

		self.centers = np.zeros((self.k, x.shape[1]))
		# Generate a uniform random sample from np.arange(x.shape[0]) of size k without replacement:
		self.centers = x[np.random.choice((np.arange(x.shape[0])), self.k, replace=False)]

		################################
		#      YOUR CODE GOES HERE     #
		################################

	def revise_centers(self, x, labels):
		"""
		it updates the centers based on the labels
		:param x: the input data of (n, m)
		:param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
		:return: updates the self.centers
		"""

		for i in range(self.k):
			wherei = np.squeeze(np.argwhere(labels == i), axis=1)
			self.centers[i, :] = x[wherei, :].mean(0)

	def predict(self, x):
		"""
		returns the labels of the input x based on the current self.centers
		:param x: input of (n, m)
		:return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
		"""
		# Source:https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
		labels = np.zeros((x.shape[0]), dtype=int)
		distance = np.zeros((x.shape[0], self.k))
		for i in range(self.k):
			row_n = np.sum((np.square((x - self.centers[i]))), axis=1)
			distance[:,i] = row_n

		# find closest cluster
		labels = np.argmin(distance, axis=1)
		##################################
		#      YOUR CODE GOES HERE       #
		##################################
		return labels

	def get_sse(self, x, labels):
		"""
		for a given input x and its cluster labels, it computes the sse with respect to self.centers
		:param x:  input of (n, m)
		:param labels: label of (n,)
		:return: float scalar of sse
		"""

		##################################
		#      YOUR CODE GOES HERE       #
		##################################
		# Source:https://books.google.com/books?id=pgyLDwAAQBAJ&pg=PA90&lpg=PA90&dq=np.linalg.norm+kmeans+sse&source=bl&ots=ZcYMXySSj8&sig=ACfU3U1zmiI6XPQa3nDeb7pNR88mwDt0TQ&hl=en&sa=X&ved=2ahUKEwiVv5ea2NzpAhXtoFsKHcogCOIQ6AEwAXoECAsQAQ#v=onepage&q=np.linalg.norm%20kmeans%20sse&f=false
		sse = 0.
		for i in range(self.k):
			sse = np.linalg.norm(x[i] - self.centers[labels[i]])

		return np.sum(np.square(sse))

	def get_purity(self, x, y):
		"""
		computes the purity of the labels (predictions) given on x by the model
		:param x: the input of (n, m)
		:param y: the ground truth class labels
		:return:
		"""
		labels = self.predict(x)
		purity = 0

		# for i in range(self.k):

		##################################
		#      YOUR CODE GOES HERE       #
		##################################
		#iterate through both predicted and truth class labels arrays, 7352 is the number of samples in the provided data
		for z in range(7352):
			if labels[z] == y[z]:
				purity = purity + 1

		purity = purity / 7352
		return purity

	def fit(self, x):
		"""
		this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
		:param x: input data of (n, m)
		:return: computes self.centers. It also returns sse_veersus_iterations for x.
		"""
		print("K is:")
		print(self.k)
		# intialize self.centers
		self.init_center(x)

		sse_vs_iter = []
		for iter in range(self.max_iter):
			# finds the cluster index for each x[i] based on the current centers
			labels = self.predict(x)

			# revises the values of self.centers based on the x and current labels
			self.revise_centers(x, labels)

			# computes the sse based on the current labels and centers.
			sse = self.get_sse(x, labels)

			sse_vs_iter.append(sse)

		return sse_vs_iter
