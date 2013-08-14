from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat
from sklearn.svm import libsvm


def plot_data(X, y, show=True):
    pos = y.nonzero()[0]
    neg = (y == 0).nonzero()[0]
    plot.plot(X[pos, 0], X[pos, 1], 'k+', markersize=7, linewidth=2)
    plot.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, linewidth=2)
    if show:
        plot.show()


def visualize_boundary_linear(X, y, model):
    support_vectors = model[1]
    coeffs = model[3]
    intercept = model[4]
    normal_vector = coeffs.dot(support_vectors).flatten()
    xp = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 100)
    yp = -(normal_vector[0] * xp + intercept) / normal_vector[1]
    plot_data(X, y, show=False)
    plot.plot(xp, yp, '-b')
    plot.show()


def visualize_boundary(X, y, model, gamma):
    plot_data(X, y, show=False)
    x1plot = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 100)
    x2plot = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    predictions = np.zeros_like(X1)
    for i in range(X1.shape[1]):
        currentX = np.require(np.vstack((X1[:, i], X2[:, i])).T, requirements='C_CONTIGUOUS')
        predictions[:, i] = libsvm.predict(
            currentX,
            support=model[0],
            SV=model[1],
            nSV=model[2],
            sv_coef=model[3],
            intercept=model[4],
            label=model[5],
            probA=model[6],
            probB=model[7],
            kernel='rbf',
            gamma=gamma
        )
    plot.contour(X1, X2, predictions, [0.0, 0.0])
    plot.show()


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-sum((x1 - x2) ** 2) / (2.0 * sigma ** 2))


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex6/ex6data1.mat')
    X = np.require(data['X'], dtype=np.float64, requirements='C_CONTIGUOUS')
    y = np.require(data['y'].flatten(), dtype=np.float64)
    plot_data(X, y)
    print('Training Linear SVM ...')
    C = 1.0
    model = libsvm.fit(X, y, kernel='linear', tol=0.001, C=C, max_iter=20)
    visualize_boundary_linear(X, y, model)
    # evaluate gaussian kernel
    x1 = np.array([1.0, 2.0, 1.0])
    x2 = np.array([0.0, 4.0, -1.0])
    sigma = 2.0
    value = gaussian_kernel(x1, x2, sigma)
    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5: %f' % value)
    print('(this value should be about 0.324652)')
    # dataset 2
    data = loadmat('../../octave/mlclass-ex6/ex6data2.mat')
    X = np.require(data['X'], dtype=np.float64, requirements='C_CONTIGUOUS')
    y = np.require(data['y'].flatten(), dtype=np.float64)
    plot_data(X, y)
    print('Training SVM with RBF Kernel ...')
    C = 1.0
    sigma = 0.1
    gamma = 1.0 / (2.0 * sigma ** 2)
    model = libsvm.fit(X, y, kernel='rbf', C=C, gamma=gamma)
    visualize_boundary(X, y, model, gamma)
