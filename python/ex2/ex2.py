import numpy as np
from matplotlib import pyplot as plot


def plot_data(X, y):
    pos = y.nonzero()[0]
    neg = (y == 0).nonzero()[0]
    plot.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=2)
    plot.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, linewidth=2)
    plot.xlabel('Exam 1 score')
    plot.ylabel('Exam 2 score')
    plot.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/mlclass-ex2/ex2data1.txt', delimiter=',')
    X = data1[:,0:2]
    # reshape to force y to be a column vector
    y = data1[:,2].reshape(-1, 1)
    m = y.size
    plot_data(X, y)
