import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat


if __name__ == '__main__':
    weights = loadmat('../../octave/mlclass-ex3/ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    print Theta1.shape
    print Theta2.shape
