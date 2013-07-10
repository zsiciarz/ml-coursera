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


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    boundary = input_layer_size * (hidden_layer_size - 1)
    Theta1 = nn_params[:boundary].reshape((hidden_layer_size - 1, input_layer_size))
    Theta2 = nn_params[boundary:].reshape((num_labels, hidden_layer_size))


if __name__ == '__main__':
    print 'Loading and Visualizing Data ...'
    data = loadmat('../../octave/mlclass-ex4/ex4data1.mat')
    X = data['X']
    y = data['y'].flatten()
    sel = np.random.permutation(X)[:100]
    display_data(sel)
    print 'Loading Saved Neural Network Parameters ...'
    weights = loadmat('../../octave/mlclass-ex4/ex4weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    m = X.shape[0]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    nn_cost_function(
        nn_params,
        input_layer_size=Theta1.shape[1],
        hidden_layer_size=Theta2.shape[1],
        num_labels=Theta2.shape[0],
        X=X,
        y=y,
        lambda_=0
    )
