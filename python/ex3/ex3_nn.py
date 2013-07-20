from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat

from ex3 import display_data, sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    A2 = sigmoid(X.dot(Theta1.T))
    A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)
    A3 = sigmoid(A2.dot(Theta2.T))
    predictions = 1 + np.argmax(A3, axis=1)
    return predictions


if __name__ == '__main__':
    print('Loading and Visualizing Data ...')
    data = loadmat('../../octave/mlclass-ex3/ex3data1.mat')
    X = data['X']
    y = data['y'].flatten()
    m = X.shape[0]
    sel = np.random.permutation(X)[:100]
    display_data(sel)
    print('Loading Saved Neural Network Parameters ...')
    weights = loadmat('../../octave/mlclass-ex3/ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    predictions = predict(Theta1, Theta2, X)
    accuracy = 100 * np.mean(predictions == y)
    print('Training set accuracy: %0.2f %%' % accuracy)
    random_X = np.random.permutation(X)
    for i in range(m):
        example = random_X[i].reshape(1, -1)
        prediction = predict(Theta1, Theta2, example)
        print('Prediction: %d (digit %d)' % (prediction, prediction % 10))
        display_data(example[:, 1:])
