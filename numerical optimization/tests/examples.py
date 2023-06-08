import numpy as np


def calc_f(x, q):
    return x.dot(q).dot(x)


def calc_g(x, q):
    return 2 * q.dot(x)


def calc_h(compute_hessian, q):
    if compute_hessian:
        return 2 * q
    return None


def quadratic_example1(x, compute_hessian=False):
    q = np.array([[1, 0], [0, 1]])
    return calc_f(x, q), calc_g(x, q), calc_h(compute_hessian, q)


def quadratic_example2(x, compute_hessian=False):
    q = np.array([[1, 0], [0, 100]])
    return calc_f(x, q), calc_g(x, q), calc_h(compute_hessian, q)


def quadratic_example3(x, compute_hessian=False):
    q_sides = np.array([[0.5 * np.sqrt(3), -0.5], [0.5, 0.5 * np.sqrt(3)]])
    q_sides_transpose = q_sides.T
    q_middle = np.array([[100, 0], [0, 1]])

    q = q_sides_transpose.dot(q_middle).dot(q_sides)
    return calc_f(x, q), calc_g(x, q), calc_h(compute_hessian, q)


def rosenbrock(x, compute_hessian=False):
    x1 = x[0]
    x2 = x[1]
    f = 100 * ((x2 - (x1 ** 2)) ** 2) + ((1 - x1) ** 2)
    g = np.array([
        -400 * x1 * (x2 - (x1 ** 2)) - 2 * (1 - x1),
        200 * (x2 - (x1 ** 2))
    ])
    h = None
    if compute_hessian:
        h = np.array([
            [-400 * (x2 - 3 * (x1 ** 2)) + 2, -400 * x1],
            [-400 * x1, 200]
        ])

    return f, g, h


def linear_function(x, compute_hessian=False):
    a = np.array([0.5,0.6])
    f = a.dot(x)
    g = a
    h = None
    if compute_hessian:
        h = [[0,0],[0,0]]
    return f, g, h


def exponential_func(x, compute_hessian=False):
    x1 = x[0]
    x2 = x[1]
    f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([
        np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1),
        3 * (np.exp(x1 + 3 * x2 - 0.1)) - 3 * (np.exp(x1 - 3 * x2 - 0.1))
    ])
    h = None
    if compute_hessian:
        h = np.array([
            [np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1),
             3 * (np.exp(x1 + 3 * x2 - 0.1)) - 3 * (np.exp(x1 - 3 * x2 - 0.1))],
            [3 * (np.exp(x1 + 3 * x2 - 0.1)) - 3 * (np.exp(x1 - 3 * x2 - 0.1)),
             9 * (np.exp(x1 + 3 * x2 - 0.1)) + 9 * (np.exp(x1 - 3 * x2 - 0.1))]
        ])
    return f, g, h
