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
    costs = (X.dot(theta) - y) ** 2
    return costs.sum() / (2.0 * m)


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = X.dot(theta)
        errors = h - y
        delta = X.transpose().dot(errors)
        theta -= (alpha / m) * delta
        J_history[i] = compute_cost(X, y, theta)
    return (theta, J_history)


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
    (theta, J_history) = gradient_descent(X, y, theta, alpha, iterations)
    print theta
    predict1 = np.array([1, 3.5]).dot(theta)
    print 'For population = 35,000, we predict a profit of %f' % (predict1[0] * 10000)
    predict2 = np.array([1, 7]).dot(theta)
    print 'For population = 70,000, we predict a profit of %f' % (predict2[0] * 10000)
