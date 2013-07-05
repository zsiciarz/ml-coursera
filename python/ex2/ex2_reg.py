import numpy as np
from matplotlib import pyplot as plot
from scipy import optimize

from ex2 import plot_data, cost_function, predict


def map_feature(X1, X2, degree=6):
    """
    Creates polynomial features up to ``degree``.
    """
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    m = X1.shape[0]
    out = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.concatenate((out, (X1 ** (i - j)) * (X2 ** j)), axis=1)
    return out


def cost_function_reg(theta, X, y, lambda_):
    """
    Regularized logistic regression cost function.
    """
    m = X.shape[0]
    cost, gradient = cost_function(theta, X, y)
    reg_cost = (lambda_ / (2.0 * m)) * np.sum(theta[1:, :] ** 2)
    reg_gradient = (lambda_ / m) * theta
    reg_gradient[0] = 0
    return cost + reg_cost, gradient + reg_gradient


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/mlclass-ex2/ex2data2.txt', delimiter=',')
    X_original = X = data1[:, 0:2]
    # reshape to force y to be a column vector
    y = data1[:, 2].reshape(-1, 1)
    plot_data(X, y)
    X = map_feature(X[:, 0], X[:, 1])
    m, n = X.shape
    initial_theta = np.zeros((n, 1))
    lambda_ = 1.0
    cost, grad = cost_function_reg(initial_theta, X, y, lambda_)
    print 'Cost at initial theta (zeros): %f' % cost
    # we need to do some wrapping and reshaping to play nice with fmin
    wrapped = lambda t: cost_function_reg(t.reshape(initial_theta.shape), X, y, lambda_)[0]
    wrapped_prime = lambda t: cost_function_reg(t.reshape(initial_theta.shape), X, y, lambda_)[1].flatten()
    result = optimize.fmin_bfgs(
        wrapped,
        initial_theta.flatten(),
        fprime=wrapped_prime,
        maxiter=400,
        full_output=True,
        disp=False
    )
    theta = result[0].reshape(initial_theta.shape)
    # plot the decision boundary
    plot_data(X_original, y, show=False)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))
    for i in range(u.size):
        for j in range(v.size):
            z[i, j] = map_feature(u[i], v[j]).dot(theta)
    plot.contour(u, v, z.T, [0.0, 0.0])
    plot.show()
    predictions = predict(theta, X)
    accuracy = 100 * np.mean(map(int, predictions == y))
    print 'Train accuracy: %0.2f %%' % accuracy
