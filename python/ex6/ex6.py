from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat
from sklearn.svm import libsvm


def plot_data(X, y, show=True):
    pos = y.nonzero()[0]
    neg = (y == 0).nonzero()[0]
    plot.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=2)
    plot.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, linewidth=2)
    if show:
        plot.show()


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex6/ex6data1.mat')
    X = np.require(data['X'], dtype=np.float64, requirements='C_CONTIGUOUS')
    y = np.require(data['y'].flatten(), dtype=np.float64)
    plot_data(X, y)
    print('Training Linear SVM ...')
    C = 1
    model = libsvm.fit(X, y, kernel='linear', tol=0.001, C=C)
