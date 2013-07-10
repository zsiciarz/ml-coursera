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


def sigmoid_gradient(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    boundary = input_layer_size * (hidden_layer_size - 1)
    Theta1 = nn_params[:boundary].reshape((hidden_layer_size - 1, input_layer_size))
    Theta2 = nn_params[boundary:].reshape((num_labels, hidden_layer_size))
    m = X.shape[0]
    possible_labels = np.arange(1, num_labels + 1)
    cost = 0.0
    for i in range(m):
        a2 = sigmoid(Theta1.dot(X[i, :]))
        a2 = np.concatenate((np.ones(1), a2))
        h = a3 = sigmoid(Theta2.dot(a2))
        y_vec = np.vectorize(int)(possible_labels == y[i])
        cost += sum(-y_vec * np.log(h) - (1.0 - y_vec) * np.log(1.0 - h))
    cost = cost / m
    # regularization
    reg_cost = (lambda_ / (2.0 * m)) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
    return cost + reg_cost


def rand_initialize_weights(L_in, L_out):
    eps = 0.12
    return np.random.uniform(-eps, eps, (L_in, L_out))


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
    input_layer_size = Theta1.shape[1]
    hidden_layer_size = Theta2.shape[1]
    num_labels = Theta2.shape[0]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    # feed-forward cost function
    cost = nn_cost_function(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0
    )
    print 'Cost at parameters (loaded from ex4weights): %f' % cost
    print '(this value should be about 0.287629)'
    # feed-forward with regularization
    print 'Checking Cost Function (w/ Regularization) ...'
    cost = nn_cost_function(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1
    )
    print 'Cost at parameters (loaded from ex4weights): %f' % cost
    print '(this value should be about 0.383770)'
    print 'Evaluating sigmoid gradient...'
    gradient = sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
    print 'Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n%s' % gradient
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
