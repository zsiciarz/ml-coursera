import numpy as np
from matplotlib import pyplot as plot

from ex2 import plot_data


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/mlclass-ex2/ex2data2.txt', delimiter=',')
    X = data1[:, 0:2]
    # reshape to force y to be a column vector
    y = data1[:, 2].reshape(-1, 1)
    plot_data(X, y)
