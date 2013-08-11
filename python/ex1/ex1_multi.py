from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot


if __name__ == '__main__':
    data = np.loadtxt('../../octave/ex1/ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = y.size
    print('First 10 examples from the dataset:')
    for i in range(10):
        print(' x = %s, y = %.0f' % (X[i], y[i]))
