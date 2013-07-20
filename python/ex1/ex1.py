from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D


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
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = X.dot(theta)
        errors = h - y
        delta = X.T.dot(errors)
        theta -= (alpha / m) * delta
        J_history[i] = compute_cost(X, y, theta)
    return (theta, J_history)


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/ex1/ex1data1.txt', delimiter=',')
    X = data1[:,0]
    y = data1[:,1]
    m = y.size
    plot_data(X, y)
    X = np.vstack((np.ones(m), X)).T
    theta = np.zeros(2)
    iterations = 1500
    alpha = 0.01
    print(compute_cost(X, y, theta))
    (theta, J_history) = gradient_descent(X, y, theta, alpha, iterations)
    print(theta)
    plot.plot(X[:,1], y, 'rx', markersize=10)
    plot.ylabel('Profit in $10,000s')
    plot.xlabel('Population of City in 10,000s')
    plot.plot(X[:,1], X.dot(theta), '-')
    plot.show()
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
    predict2 = np.array([1, 7]).dot(theta)
    print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))
    print('Visualizing J(theta_0, theta_1) ...')
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t)
    plot.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
    plot.show()
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    t0, t1 = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(t0, t1, J_vals)
    plot.show()
