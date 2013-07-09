import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat

from ex3 import display_data


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex3/ex3data1.mat')
    X = data['X']
    y = data['y'].flatten()
    sel = np.random.permutation(X)[:100]
    display_data(sel)
    weights = loadmat('../../octave/mlclass-ex3/ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    print Theta1.shape
    print Theta2.shape
