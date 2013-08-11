from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot

from ex1 import gradient_descent


def normalize_features(X, mu=None, sigma=None):
    m = X.shape[0]
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0, ddof=1)
    # don't change the intercept term
    mu[0] = 0.0
    sigma[0] = 1.0
    for i in range(m):
        X[i, :] = (X[i, :] - mu) / sigma
    return X, mu, sigma


if __name__ == '__main__':
    data = np.loadtxt('../../octave/ex1/ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = X.shape[0]
    print('First 10 examples from the dataset:')
    for i in range(10):
        print(' x = %s, y = %.0f' % (X[i], y[i]))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # normalize features
    X, mu, sigma = normalize_features(X)
    print('Running gradient descent ...')
    alpha = 0.01
    num_iters = 400
    theta = np.zeros(3)
    [theta, J_history] = gradient_descent(X, y, theta, alpha, num_iters)
    print('Theta computed from gradient descent:\n%s' % theta)
    plot.plot(J_history, '-b')
    plot.xlabel('Number of iterations')
    plot.ylabel('Cost J')
    plot.show()
