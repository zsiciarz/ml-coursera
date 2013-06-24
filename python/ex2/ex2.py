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


def cost_function(theta, X, y):
    cost = 0
    m = X.shape[0]
    grad = np.zeros_like(theta)
    for i in range(m):
        h = sigmoid(theta.transpose().dot(X[i, :].transpose()))
        cost += (-y[i] * np.log(h) - (1.0 - y[i]) * np.log(1 - h))
        for j in range(theta.size):
            grad[j] = grad[j] + (h - y[i]) * X[i, j]
    return (cost / m, grad / m)


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/mlclass-ex2/ex2data1.txt', delimiter=',')
    X = data1[:, 0:2]
    # reshape to force y to be a column vector
    y = data1[:, 2].reshape(-1, 1)
    plot_data(X, y)
    m, n = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    initial_theta = np.zeros((n + 1, 1))
    cost, grad = cost_function(initial_theta, X, y)
    print 'Cost at initial theta (zeros): %f' % cost
    print 'Gradient at initial theta (zeros):'
    print grad
