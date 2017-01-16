from __future__ import print_function

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plot
import matplotlib.cm as cm
from scipy import optimize
from scipy.io import loadmat


def display_data(X):
    """
    Transforms each input row into a rectangular image part and plots
    the resulting image.
    """
    m, n = X.shape
    example_width = int(np.around(np.sqrt(n)))
    example_height = int(n / example_width)
    display_rows = int(np.sqrt(m))
    display_cols = int(m / display_rows)
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

def nn_cost_function_vectorized(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    boundary = (input_layer_size + 1) * hidden_layer_size
    Theta1 = nn_params[:boundary].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[boundary:].reshape((num_labels, hidden_layer_size + 1))
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)
    m = X.shape[0]

    #forward propagation
    y = np.reshape(y, (np.shape(y)[0],1))
    yMatrix = 1*(np.arange(1,num_labels+1) == y)
    a1 = X
    z2 = np.dot(a1,Theta1.T)
    a2 = np.concatenate((np.ones((m,1)), sigmoid(z2)), axis=1)
    z3 = np.dot(a2,Theta2.T)
    a3 = sigmoid(z3)

    #cost function
    J = -np.sum(yMatrix*np.log(a3)+(1-yMatrix)*np.log(1.0-a3))/m
    J = J + lambda_/(2*m)*(np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))

    #back propagation
    d3 = a3-yMatrix
    d2 = np.dot(Theta2[:,1:].T, d3.T)*sigmoid_gradient(z2).T
    Delta1 = np.dot(d2, a1)
    Delta2 = np.dot(d3.T, a2)

    #gradient calculation
    Theta1_reg = lambda_/m*Theta1
    Theta2_reg = lambda_/m*Theta2
    Theta1_reg[:,0] = 0
    Theta2_reg[:,0] = 0
    Theta1_grad = 1/m*Delta1 + Theta1_reg
    Theta2_grad = 1/m*Delta2 + Theta2_reg
    gradient = np.concatenate((Theta1_grad.flatten('C'), Theta2_grad.flatten('C')))
    
    return J, gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    boundary = (input_layer_size + 1) * hidden_layer_size
    Theta1 = nn_params[:boundary].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[boundary:].reshape((num_labels, hidden_layer_size + 1))
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)
    m = X.shape[0]
    possible_labels = np.arange(1, num_labels + 1)
    cost = 0.0
    for i in range(m):
        a1 = X[i, :]
        a2 = sigmoid(Theta1.dot(a1))
        a2 = np.concatenate((np.ones(1), a2))
        h = a3 = sigmoid(Theta2.dot(a2))
        y_vec = np.vectorize(int)(possible_labels == y[i])
        cost += sum(-y_vec * np.log(h) - (1.0 - y_vec) * np.log(1.0 - h))
        # backprop
        delta3 = a3 - y_vec
        Theta2_grad += np.outer(delta3, a2)
        delta2 = Theta2.T.dot(delta3) * a2 * (1 - a2)
        Theta1_grad += np.outer(delta2[1:], a1)
    cost = cost / m
    Theta1_grad /= m
    Theta2_grad /= m
    # regularization
    reg_cost = (lambda_ / (2.0 * m)) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
    Theta1_grad += (lambda_ / m) * np.concatenate((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]), axis=1)
    Theta2_grad += (lambda_ / m) * np.concatenate((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]), axis=1)
    gradient = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return cost + reg_cost, gradient


def rand_initialize_weights(L_in, L_out):
    eps = 0.12
    return np.random.uniform(-eps, eps, (L_out, 1 + L_in))


def debug_initialize_weights(fan_out, fan_in):
    return np.sin(1 + np.arange((1 + fan_in) * fan_out)).reshape((1 + fan_in, fan_out)).T / 10.0


def check_nn_gradients(lambda_=0.0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generate some 'random' test data
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    y = 1 + np.arange(1, m + 1) % num_labels
    # unroll parameters
    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    cost_func = lambda p: nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
    cost, grad = cost_func(nn_params)
    num_grad = compute_numerical_gradient(cost_func, nn_params)
    print(np.vstack((grad, num_grad)).T)
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    diff = norm(num_grad - grad) / norm(num_grad + grad)
    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %g' % diff)


def compute_numerical_gradient(cost_func, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    eps = 1e-4
    for p in range(theta.size):
        perturb[p] = eps
        loss1, _ = cost_func(theta - perturb)
        loss2, _ = cost_func(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2.0 * eps)
        perturb[p] = 0.0
    return numgrad


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    h1 = sigmoid(X.dot(Theta1.T))
    h2 = sigmoid(np.concatenate((np.ones((m, 1)), h1), axis=1).dot(Theta2.T))
    return 1 + np.argmax(h2, axis=1)


if __name__ == '__main__':
    print('Loading and Visualizing Data ...')
    data = loadmat('../../octave/mlclass-ex4/ex4data1.mat')
    X = data['X']
    y = data['y'].flatten()
    sel = np.random.permutation(X)[:100]
    #display_data(sel)
    print('Loading Saved Neural Network Parameters ...')
    weights = loadmat('../../octave/mlclass-ex4/ex4weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    m = X.shape[0]
    input_layer_size = Theta1.shape[1] - 1
    hidden_layer_size = Theta2.shape[1] - 1
    num_labels = Theta2.shape[0]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
    # feed-forward cost function
    cost, grad = nn_cost_function(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0
    )
    print('Cost at parameters (loaded from ex4weights): %f' % cost)
    print('(this value should be about 0.287629)')
    # feed-forward with regularization
    print('Checking Cost Function (w/ Regularization) ...')
    cost, grad = nn_cost_function(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1
    )
    print('Cost at parameters (loaded from ex4weights): %f' % cost)
    print('(this value should be about 0.383770)')
    print('Evaluating sigmoid gradient...')
    gradient = sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n%s' % gradient)
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
    print('Checking Backpropagation...')
    check_nn_gradients()
    print('Checking Backpropagation (w/ Regularization) ...')
    lambda_ = 3.0
    check_nn_gradients(lambda_)
    cost, grad = nn_cost_function(
        nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_
    )
    print('Cost at (fixed) debugging parameters (w/ lambda = 3): %f ' % cost)
    print('(this value should be about 0.576051)')
    print('Training Neural Network...')
    result = optimize.minimize(
        nn_cost_function,
        initial_nn_params,
        args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_),
        method='CG',
        jac=True,
        options={
            'maxiter': 50,
            'disp': False,
        }
    )
    nn_params = result.x
    boundary = (input_layer_size + 1) * hidden_layer_size
    Theta1 = nn_params[:boundary].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[boundary:].reshape((num_labels, hidden_layer_size + 1))
    print('Visualizing Neural Network ...')
    #display_data(Theta1[:, 1:])
    predictions = predict(Theta1, Theta2, X)
    accuracy = 100 * np.mean(predictions == y)
    print('Training set accuracy: %0.2f %%' % accuracy)
