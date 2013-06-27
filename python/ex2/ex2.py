import numpy as np
from matplotlib import pyplot as plot
from scipy import optimize


def plot_data(X, y, show=True):
    pos = y.nonzero()[0]
    neg = (y == 0).nonzero()[0]
    plot.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=2)
    plot.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, linewidth=2)
    plot.xlabel('Exam 1 score')
    plot.ylabel('Exam 2 score')
    if show:
        plot.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cost_function(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    cost = sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
    grad = X.transpose().dot(h - y)
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
    # we need to do some wrapping and reshaping to play nice with fmin
    wrapped = lambda t: cost_function(t.reshape(initial_theta.shape), X, y)[0]
    result = optimize.fmin(
        wrapped,
        initial_theta.flatten(),
        maxiter=400,
        full_output=True,
        disp=False
    )
    theta = result[0].reshape(initial_theta.shape)
    cost = result[1]
    print 'Cost at theta found by scipy.optimize.fmin: %f' % cost
    print 'theta:'
    print theta
    # plot the decision boundary
    plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
    plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
    plot_data(X[:, 1:], y, show=False)
    plot.plot(plot_x, plot_y)
    plot.show()
