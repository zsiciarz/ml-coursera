from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat
import svmutil


def plot_data(X, y, show=True):
    pos = y.nonzero()[0]
    neg = (y == 0).nonzero()[0]
    plot.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=2)
    plot.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, linewidth=2)
    if show:
        plot.show()


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex6/ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)
    print('Training Linear SVM ...')
    C = 1
    params = '-q -c %d -t %d -e %f' % (
        C,  # cost
        0,  # linear kernel
        0.001,  # epsilon
    )
    model = svmutil.svm_train(y.tolist(), X.tolist(), params)
