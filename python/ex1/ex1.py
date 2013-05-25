import numpy as np
from matplotlib import pyplot as plot


def warmup_exercise():
    return np.identity(5)


def plot_data(X, y):
    plot.plot(X, y, 'rx', markersize=10)
    plot.ylabel('Profit in $10,000s')
    plot.xlabel('Population of City in 10,000s')
    plot.show()


if __name__ == '__main__':
    data1 = np.loadtxt('../../octave/ex1/ex1data1.txt', delimiter=',')
    X = data1[:,0]
    y = data1[:,1]
    m = y.size
    plot_data(X, y)
