import numpy as np
#import copy


class PCA():
	"""
	PCA. A class to reduce dimensions
	"""

	def __init__(self, retain_ratio):
		"""

		:param retain_ratio: percentage of the variance we maitain (see slide for definition)
		"""
		self.retain_ratio = retain_ratio

	@staticmethod
	def mean(x):
		"""
		returns mean of x
		:param x: matrix of shape (n, m)
		:return: mean of x of with shape (m,)
		"""
		return x.mean(axis=0)

	@staticmethod
	def cov(x):
		"""
		returns the covariance of x,
		:param x: input data of dim (n, m)
		:return: the covariance matrix of (m, m)
		"""
		return np.cov(x.T)

	@staticmethod
	def eig(c):
		"""
		returns the eigval and eigvec
		:param c: input matrix of dim (m, m)
		:return:
			eigval: a numpy vector of (m,)
			eigvec: a matrix of (m, m), column ``eigvec[:,i]`` is the eigenvector corresponding to the
		eigenvalue ``eigval[i]``
			Note: eigval is not necessarily ordered
		"""

		eigval, eigvec = np.linalg.eig(c)
		eigval = np.real(eigval)
		eigvec = np.real(eigvec)
		return eigval, eigvec


	def fit(self, x):
		"""
		fits the data x into the PCA. It results in self.eig_vecs and self.eig_values which will
		be used in the transform method
		:param x: input data of shape (n, m); n instances and m features
		:return:
			sets proper values for self.eig_vecs and eig_values
		"""

		x = x - PCA.mean(x) # TODO Not needed? Meh, keep it anyway

		# Get eigenvalues and eigenvectors
		self.eig_vals, self.eig_vecs = PCA.eig(PCA.cov(x))

		# Pair eigenvalues and eigenvectors in tuples (eigenpairs)
		n = len(self.eig_vals)
		pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:,i]) for i in range(n)]
		# Sort by first column, descending order
		pairs.sort(key=lambda x: x[0], reverse=True)
		# Split back out
		self.eig_vals = [x[0] for x in pairs]
		self.eig_vecs = [x[1] for x in pairs]

		# Determine dimensional reduction from retain ratio
		sumd = np.sum(self.eig_vals) # Variance of original data
		sumk = 0 # For accumulating first k eigenvalues
		k = -1 # For tracking reduced dimensionality
		for i in range(n):
			sumk += self.eig_vals[i]
			if sumk >= self.retain_ratio * sumd:
				k = i + 1 # Criterion met; use this many dimensions
				break

		# Keep only the first k eigenpairs
		self.eig_vals = self.eig_vals[0:k]
		self.eig_vecs = np.array(self.eig_vecs[0:k]).T


	def transform(self, x):
		"""
		projects x into lower dimension based on current eig_vals and eig_vecs
		:param x: input data of shape (n, m)
		:return: projected data with shape (n, len of eig_vals)
		"""

		if isinstance(x, np.ndarray):
			x = np.asarray(x)
		if self.eig_vecs is not None:
			return np.matmul(x, self.eig_vecs)
		else:
			return x
