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
            'maxiter': 200,
            'disp': False,
        }
    )
    theta = result.x
    return theta


def learning_curve(X, y, Xval, yval, lambda_):
    m = X.shape[0]
    err_train = np.zeros(m + 1)
    err_val = np.zeros(m + 1)
    for i in range(1, m + 1):
        Xtrain = X[:i, :]
        ytrain = y[:i]
        theta = train_linear_regression(Xtrain, ytrain, lambda_)
        err_train[i] = cost_function_reg(theta, Xtrain, ytrain, 0.0)[0]
        err_val[i] = cost_function_reg(theta, Xval, yval, 0.0)[0]
    return err_train, err_val


def poly_features(X, power=8):
    """
    Creates polynomial features up to ``power``.
    """
    cols = [X ** p for p in range(power + 1)]
    return np.vstack(cols).T


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


def plot_fit(min_x, max_x, mu, sigma, theta, power):
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    Xpoly = poly_features(x, power)
    Xpoly, _, _ = normalize_features(Xpoly, mu, sigma)
    plot.plot(x, Xpoly.dot(theta), '--')


def validation_curve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros_like(lambda_vec)
    error_val = np.zeros_like(lambda_vec)
    for i, lambda_ in enumerate(lambda_vec):
        theta = train_linear_regression(X, y, lambda_)
        error_train[i] = cost_function_reg(theta, X, y, 0.0)[0]
        error_val[i] = cost_function_reg(theta, Xval, yval, 0.0)[0]
    return lambda_vec, error_train, error_val


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
    plot.plot(range(m + 1), err_train)
    plot.plot(range(m + 1), err_val)
    plot.title('Learning curve for linear regression')
    plot.xlabel('Number of training examples')
    plot.ylabel('Error')
    plot.show()
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m + 1):
        print('  \t%d\t\t%f\t%f' % (i, err_train[i], err_val[i]))
    # map polynomial features
    power = 8
    Xpoly = poly_features(X[:, 1], power)
    Xpoly, mu, sigma = normalize_features(Xpoly)
    Xval_poly = poly_features(Xval[:, 1], power)
    Xval_poly, _, _ = normalize_features(Xval_poly)
    print('Normalized Training Example 1:\n%s' % Xpoly[0, :])
    # train linear regression with polynomial features
    lambda_ = 0.0
    theta = train_linear_regression(Xpoly, y, lambda_)
    plot.plot(X[:, 1], y, 'rx', markersize=10)
    plot_fit(X.min(), X.max(), mu, sigma, theta, power)
    plot.xlabel('Change in water level (x)')
    plot.ylabel('Water flowing out of the dam (y)')
    plot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
    plot.show()
    # learning curve for polynomial fit
    err_train, err_val = learning_curve(Xpoly, y, Xval_poly, yval, lambda_)
    plot.plot(range(m + 1), err_train)
    plot.plot(range(m + 1), err_val)
    plot.title('Polynomial Regression Learning Curve (lambda = %f)' % (lambda_))
    plot.xlabel('Number of training examples')
    plot.ylabel('Error')
    plot.show()
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m + 1):
        print('  \t%d\t\t%f\t%f' % (i, err_train[i], err_val[i]))
    # lambda selection
    lambda_vec, err_train, err_val = validation_curve(Xpoly, y, Xval_poly, yval)
    plot.plot(lambda_vec, err_train)
    plot.plot(lambda_vec, err_val)
    plot.xlabel('lambda')
    plot.ylabel('Error')
    plot.show()
    print('# lambda\tTrain Error\tCross Validation Error')
    for i, lambda_ in enumerate(lambda_vec):
        print('  \t%f\t\t%f\t%f' % (lambda_, err_train[i], err_val[i]))
