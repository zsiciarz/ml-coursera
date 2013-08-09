from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy import optimize
from scipy.io import loadmat


if __name__ == '__main__':
    print('Loading and Visualizing Data ...')
    data = loadmat('../../octave/mlclass-ex5/ex5data1.mat')
    X = data['X']
    y = data['y'].flatten()
    plot.plot(X, y, 'rx', markersize=10)
    plot.show()
