from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot

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
    m = y.size
    print('First 10 examples from the dataset:')
    for i in range(10):
        print(' x = %s, y = %.0f' % (X[i], y[i]))
