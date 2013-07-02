import numpy as np
from matplotlib import pyplot as plot
import matplotlib.cm as cm
from scipy.io import loadmat


def display_data(X):
    """
    TODO: reshape the data to look like 20x20 images
    """
    plot.imshow(X, cm.Greys)
    plot.show()


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex3/ex3data1.mat')
    X = data['X']
    y = data['y']
    sel = np.random.permutation(X)[:100]
    display_data(sel)
