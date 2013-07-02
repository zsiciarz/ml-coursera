import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex3/ex3data1.mat')
    X = data['X']
    y = data['y']
    sel = np.random.permutation(X)
