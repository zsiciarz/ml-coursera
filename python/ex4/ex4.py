import numpy as np
from matplotlib import pyplot as plot
import matplotlib.cm as cm
from scipy.io import loadmat


def display_data(X):
    """
    Transforms each input row into a rectangular image part and plots
    the resulting image.
    """
    m, n = X.shape
    example_width = int(np.around(np.sqrt(n)))
    example_height = n / example_width
    display_rows = int(np.sqrt(m))
    display_cols = m / display_rows
    display_array = np.ones((
        display_rows * example_height, display_cols * example_width
    ))
    for i in range(display_rows):
        for j in range(display_cols):
            idx = i * display_cols + j
            image_part = X[idx, :].reshape((example_height, example_width))
            display_array[
                (j * example_height):((j + 1) * example_height),
                (i * example_width):((i + 1) * example_width)
            ] = image_part
    plot.imshow(display_array.T, cm.Greys)
    plot.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    data = loadmat('../../octave/mlclass-ex4/ex4data1.mat')
    X = data['X']
    y = data['y'].flatten()
    sel = np.random.permutation(X)[:100]
    display_data(sel)
