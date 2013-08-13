from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex6/ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
