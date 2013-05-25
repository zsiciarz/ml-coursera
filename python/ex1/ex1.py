import numpy as np
from matplotlib import pyplot as plot


def warmup_exercise():
    return np.identity(5)


def plot_data(X, y):
    plot.plot(X, y, 'rx', markersize=10)
    plot.ylabel('Profit in $10,000s')
    plot.xlabel('Population of City in 10,000s')
    plot.show()


def compute_cost(X, y, theta):
    m = y.size
    cost = 0.0
    for i in range(m):
        cost += (np.dot(theta.transpose(), X[i]) - y[i]) ** 2
    return cost / (2.0 * m)


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/ex1/ex1data1.txt', delimiter=',')
    # reshape to force X and y to be column vectors
    X = data1[:,0].reshape(-1, 1)
    y = data1[:,1].reshape(-1, 1)
    m = y.size
    plot_data(X, y)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    print compute_cost(X, y, theta)
