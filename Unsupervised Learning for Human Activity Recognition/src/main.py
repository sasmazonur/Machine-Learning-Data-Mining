import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

sns.set()

import argparse

from utils import load_data
from decompose import PCA
from clustering import KMeans


def load_args():
	parser = argparse.ArgumentParser(description='arguments')

	parser.add_argument('--pca', default=0, type=int,
						help='set to 1 to run pca')
	parser.add_argument('--kmeans', default=0, type=int,
						help='set to 1 to run kmeans')
	parser.add_argument("--pcameans", default=0, type=int,
						help="set to 1 to run kmeans on pca")
	parser.add_argument("--all", default=1, type=int,
						help="set to 1 to run everything")

	parser.add_argument('--pca_retain_ratio', default=.9, type=float)
	parser.add_argument('--kmeans_max_k', default=15, type=int)
	parser.add_argument('--kmeans_max_iter', default=20, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)

	args = parser.parse_args()

	return args


def plot_y_vs_x_list(y_vs_x, x_label, y_label, save_path):
	fld = os.path.join(args.root_dir, save_path)
	if not os.path.exists((fld)):
		os.mkdir(fld)

	plots_per_fig = 2

	ks_sses_keys = list(range(0, len(y_vs_x)))
	js = list(range(0, len(ks_sses_keys), plots_per_fig))

	for j in js:
		pp = ks_sses_keys[j:j + plots_per_fig]
		fig = plt.figure(constrained_layout=True)
		gs = gridspec.GridSpec(len(pp), 1, figure=fig)
		i = 0
		for k in pp:
			ax = fig.add_subplot(gs[i, :])
			ax.set_ylabel('%s (k=%d)' % (y_label, k))
			ax.set_xlabel(x_label)
			ax.plot(range(1, len(y_vs_x[k]) + 1), [x for x in y_vs_x[k]], linewidth=2)
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			i += 1

		fig.savefig(os.path.join(fld, '%d_%d.png' % (pp[0], pp[-1])))

	print('Saved at : %s' % fld)


def plot_y_vs_x(ys_vs_x, x_label, y_label, save_path):
	fld = os.path.join(args.root_dir, save_path)
	if not os.path.exists((fld)):
		os.mkdir(fld)

	fig = plt.figure(constrained_layout=True)
	gs = gridspec.GridSpec(1, 1, figure=fig)
	ax = fig.add_subplot(gs[0, :])
	ax.set_ylabel(y_label)
	ax.set_xlabel(x_label)
	ax.plot(range(1, len(ys_vs_x) + 1), ys_vs_x, linewidth=2)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	fig.savefig(os.path.join(fld, 'plot.png'))

	print('Saved at : %s' % fld)


def visualize(x, y, r=None, d=None):
	for class_label in np.unique(y):
		plt.scatter(x[y==class_label,0], x[y==class_label,1], s=5, alpha=0.2, label=class_label)
	plt.legend(fancybox = True, title = "Class", markerscale = 2, shadow = True)
	if r == None or d == None:
		plt.title("PCA Transformation", fontdict = {'fontsize' : 25})
	else:
		plt.title("PCA: d = {} for r = {}".format(d, r), fontdict = {'fontsize' : 25})
	plt.xlabel("Principal Dimension 1")
	plt.ylabel("Principal Dimension 2")
	plt.show()


# Plot the average (over 5 runs) of SSE versus iterations for k = 6.
def apply_kmeans(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
	print('kmeans\n')
	train_sses_vs_iter = []
	train_sses_vs_k = []
	train_purities_vs_k = []

	##################################
	#      YOUR CODE GOES HERE       #
	##################################
	# iterations for 5 different runs of k-means.
	for k in range(0, 5):
		kmeans = KMeans(6, kmeans_max_iter)
		sse_vs_iter = kmeans.fit(x_train)
		train_sses_vs_iter.append(sse_vs_iter)
		train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
		train_sses_vs_k.append(min(sse_vs_iter))
		if k == 0:
			avg_list = [0] * len(sse_vs_iter)
		avg_list = [avg_list[i] + sse_vs_iter[i] for i in range(len(sse_vs_iter))]


	plot_y_vs_x_list(train_sses_vs_iter, x_label='iter', y_label='sse',
					 save_path='plot_sse_vs_k_subplots_%d'%do_pca)
	plot_y_vs_x(avg_list, x_label='iterations', y_label='sse',
				save_path='plot_sse_vs_iter_%d'%do_pca)

# def apply_kmeans(do_pca, x_train, y_train, x_test, y_test, kmeans_max_iter, kmeans_max_k):
# Plot the average (over 5 runs) of the SSE versus k for k ∈ 1...10.
def apply_kmeans_2(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
	print('kmeans\n')
	train_sses_vs_iter = []
	train_sses_vs_k = []
	train_purities_vs_k = []
	avg_me =[]

	for k in range(1,11):
		for it in range(0,5):
			kmeans = KMeans(k, kmeans_max_iter)
			sse_vs_iter = kmeans.fit(x_train)
			train_sses_vs_iter.append(sse_vs_iter)
			train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
			train_sses_vs_k.append(min(sse_vs_iter))
		avg_me.append((sum(train_sses_vs_k)/len(train_sses_vs_k)))

	plot_y_vs_x(avg_me, x_label='k', y_label='sse',
				save_path='plot_sse_vs_k_%d'%do_pca)

# Plot the average of purity versus k for k ∈ 1 . . . 10 for the train set and make observation on this.
def apply_kmeans_3(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
	train_sses_vs_iter = []
	train_sses_vs_k = []
	train_purities_vs_k = []
	averg_list = []

	for k in range(1, 11):
		for it in range(0,5):
			kmeans = KMeans(k, kmeans_max_iter)
			sse_vs_iter = kmeans.fit(x_train)
			train_sses_vs_iter.append(sse_vs_iter)
			train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
			train_sses_vs_k.append(min(sse_vs_iter))
		averg_list.append((sum(train_purities_vs_k)/len(train_purities_vs_k)))
	#plot the average purity
	plot_y_vs_x(averg_list, x_label='k', y_label='purities',
				save_path='plot_purity_vs_k_%d'%do_pca)



if __name__ == '__main__':

	args = load_args()
	x_train, y_train = load_data(args.root_dir)

	if args.all == 1 or args.pca == 1:

		r = args.pca_retain_ratio

		pca = PCA(r)
		pca.fit(x_train)
		d = len(pca.eig_vals)
		x_trans = pca.transform(x_train)
		visualize(x_trans, y_train, pca.retain_ratio, d)
		print("For r = {}, d = {}".format(pca.retain_ratio, d))

		if args.all == 1 or args.pcameans == 1:
			pass ## TODO part 2-4: apply k-means for k = 1:10 on x_trans, plot purity, optimize r

	if args.kmeans == 1:
		apply_kmeans(args.kmeans, x_train, y_train, args.kmeans_max_iter, args.kmeans_max_k)
		apply_kmeans_2(args.kmeans, x_train, y_train, args.kmeans_max_iter, args.kmeans_max_k)
		apply_kmeans_3(args.kmeans, x_train, y_train, args.kmeans_max_iter, args.kmeans_max_k)
		# apply_kmeans(args.pca, x_train, y_train, x_test, y_test, args.kmeans_max_iter, args.kmeans_max_k)

	print('Done')
