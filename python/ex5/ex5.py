from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy import optimize
from scipy.io import loadmat


def cost_function(theta, X, y):
    """
    Linear regression cost function.
    """
    m = y.size
    h = X.dot(theta)
    errors = h - y
    cost = sum(errors ** 2) / (2.0 * m)
    gradient = (1.0 / m) * X.T.dot(errors)
    return cost, gradient


def cost_function_reg(theta, X, y, lambda_):
    """
    Regularized linear regression cost function.
    """
    m = X.shape[0]
    cost, gradient = cost_function(theta, X, y)
    reg_cost = (lambda_ / (2.0 * m)) * np.sum(theta[1:] ** 2)
    reg_gradient = (lambda_ / m) * theta
    reg_gradient[0] = 0
    return cost + reg_cost, gradient + reg_gradient


def train_linear_regression(X, y, lambda_):
    n = X.shape[1]
    initial_theta = np.zeros(n)
    result = optimize.minimize(
        cost_function_reg,
        initial_theta,
        args=(X, y, lambda_),
        method='CG',
        jac=True,
        options={
            'maxiter': 400,
            'disp': False,
        }
    )
    theta = result.x
    return theta


def learning_curve(X, y, Xval, yval, lambda_):
    m = X.shape[0]
    err_train = np.zeros_like(y)
    err_val = np.zeros_like(y)
    for i in range(1, m):
        Xtrain = X[:i, :]
        ytrain = y[:i]
        theta = train_linear_regression(Xtrain, ytrain, lambda_)
        err_train[i] = cost_function_reg(theta, Xtrain, ytrain, 0.0)[0]
        err_val[i] = cost_function_reg(theta, Xval, yval, 0.0)[0]
    return err_train, err_val


if __name__ == '__main__':
    print('Loading and Visualizing Data ...')
    data = loadmat('../../octave/mlclass-ex5/ex5data1.mat')
    X = data['X']
    y = data['y'].flatten()
    Xval = data['Xval']
    yval = data['yval'].flatten()
    m = X.shape[0]
    plot.plot(X, y, 'rx', markersize=10)
    plot.xlabel('Change in water level (x)')
    plot.ylabel('Water flowing out of the dam (y)')
    plot.show()
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    Xval = np.concatenate((np.ones((Xval.shape[0], 1)), Xval), axis=1)
    theta = np.array([1.0, 1.0])
    # linear regression cost and gradient
    cost, gradient = cost_function_reg(theta, X, y, 1.0)
    print('Cost at theta = [1, 1]: %f' % cost)
    print('(this value should be about 303.993192)')
    print('Gradient at theta = [1, 1]: \n %s' % gradient)
    print('(this value should be about [-15.303016, 598.250744])')
    # training linear regression
    lambda_ = 0.0
    theta = train_linear_regression(X, y, lambda_)
    plot.plot(X[:, 1], y, 'rx', markersize=10)
    plot.plot(X[:, 1], X.dot(theta), '--')
    plot.xlabel('Change in water level (x)')
    plot.ylabel('Water flowing out of the dam (y)')
    plot.show()
    # learning curve
    lambda_ = 0.0
    err_train, err_val = learning_curve(X, y, Xval, yval, lambda_)
    plot.plot(range(m), err_train)
    plot.plot(range(m), err_val)
    plot.show()
