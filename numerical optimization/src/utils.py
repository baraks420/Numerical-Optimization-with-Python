import matplotlib.pyplot as plt
import numpy as np


def function_transformer(f, x, y):
    res = f(np.array([x, y]))[0]
    return res


def plot_contour(f, xlimit, ylimit, level=4, algorithm_paths=None, title=""):
    x_list = np.linspace(xlimit[0], xlimit[1], (xlimit[1] - xlimit[0]) + 50)
    y_list = np.linspace(ylimit[0], ylimit[1], (ylimit[1] - ylimit[0]) + 50)
    x, y = np.meshgrid(x_list, y_list)
    transformer = np.vectorize(function_transformer)
    values = transformer(f, x, y)
    fig, ax = plt.subplots(1, 1)
    cb = ax.contourf(x, y, values, level)
    fig.colorbar(cb)
    ax.set_title(f"contour of a {title}")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if algorithm_paths is not None:
        for path, label in algorithm_paths:
            x = path[0]
            y = path[1]
            plt.plot(x, y, label=label)
    plt.legend()
    plt.show()


def plot_value_iterations(methods, name_of_function=""):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"value per iteration - {name_of_function}")
    ax.set_xlabel('iteration number')
    ax.set_ylabel('value')
    for value, label in methods:
        _ = ax.plot(range(len(value)), value, label=label)
    plt.legend()
    plt.show()
