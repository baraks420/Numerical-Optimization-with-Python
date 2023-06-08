import unittest
import tests.examples as ex
from src import utils, unconstrained_min
import numpy as np


class TestMinimizer(unittest.TestCase):

    def testCircles(self):
        f = ex.quadratic_example1
        name = "circles"
        minimizer = unconstrained_min.LineSearchMinimization(f)
        minimizer.gradient_descent()
        path_gd, value_gd = minimizer.path, minimizer.values
        minimizer.newton()
        path_new, value_new = minimizer.path, minimizer.values
        minimizer.SR1()
        path_sr1, value_sr1 = minimizer.path, minimizer.values
        minimizer.BFGS()
        path_bfgs, value_bfgs = minimizer.path, minimizer.values
        utils.plot_value_iterations(name_of_function=name, methods=[(value_gd, "gradient descent"),
                                                                    (value_new, "newton"),
                                                                    (value_sr1, "SR1"),
                                                                    (value_bfgs, "BFGS")
                                                                    ])
        utils.plot_contour(f, [-2, 2], [-2, 2], title=name, level=20, algorithm_paths=[(path_gd, "gradient descent"),
                                                                                       (path_new, "newton"),
                                                                                       (path_sr1, "SR1"),
                                                                                       (path_bfgs, "BFGS")
                                                                                       ])

    def testEllipses(self):
        f = ex.quadratic_example2
        name = " axis aligned ellipses"
        minimizer = unconstrained_min.LineSearchMinimization(f)
        minimizer.gradient_descent()
        path_gd, value_gd = minimizer.path, minimizer.values
        minimizer.newton()
        path_new, value_new = minimizer.path, minimizer.values
        minimizer.SR1()
        path_sr1, value_sr1 = minimizer.path, minimizer.values
        minimizer.BFGS()
        path_bfgs, value_bfgs = minimizer.path, minimizer.values
        utils.plot_value_iterations(name_of_function=name, methods=[(value_gd, "gradient descent"),
                                                                    (value_new, "newton"),
                                                                    (value_sr1, "SR1"),
                                                                    (value_bfgs, "BFGS")
                                                                    ])
        utils.plot_contour(f, [-2, 2], [-2, 2], title=name, level=20, algorithm_paths=[(path_gd, "gradient descent"),
                                                                                       (path_new, "newton"),
                                                                                       (path_sr1, "SR1"),
                                                                                       (path_bfgs, "BFGS")
                                                                                       ])

    def testRotatedEllipses(self):
        f = ex.quadratic_example3
        name = "rotated ellipses"
        minimizer = unconstrained_min.LineSearchMinimization(f)
        minimizer.gradient_descent()
        path_gd, value_gd = minimizer.path, minimizer.values
        minimizer.newton()
        path_new, value_new = minimizer.path, minimizer.values
        minimizer.SR1()
        path_sr1, value_sr1 = minimizer.path, minimizer.values
        minimizer.BFGS()
        path_bfgs, value_bfgs = minimizer.path, minimizer.values
        utils.plot_value_iterations(name_of_function=name, methods=[(value_gd, "gradient descent"),
                                                                    (value_new, "newton"),
                                                                    (value_sr1, "SR1"),
                                                                    (value_bfgs, "BFGS")
                                                                    ])
        utils.plot_contour(f, [-2, 2], [-2, 2], title=name, level=20, algorithm_paths=[(path_gd, "gradient descent"),
                                                                                       (path_new, "newton"),
                                                                                       (path_sr1, "SR1"),
                                                                                       (path_bfgs, "BFGS")
                                                                                       ])

    def testRosenbrock(self):
        f = ex.rosenbrock
        name = "rosenbrock"
        minimizer_gd = unconstrained_min.LineSearchMinimization(f=f, x0=np.array([-1, 2]), max_iter=10000)
        minimizer = unconstrained_min.LineSearchMinimization(f=f, x0=np.array([-1, 2]))
        minimizer_gd.gradient_descent()
        path_gd, value_gd = minimizer_gd.path, minimizer_gd.values
        minimizer.newton()
        path_new, value_new = minimizer.path, minimizer.values
        minimizer.SR1()
        path_sr1, value_sr1 = minimizer.path, minimizer.values
        minimizer.BFGS()
        path_bfgs, value_bfgs = minimizer.path, minimizer.values
        utils.plot_value_iterations(name_of_function=name, methods=[(value_gd, "gradient descent"),
                                                                    (value_new, "newton"),
                                                                    (value_sr1, "SR1"),
                                                                    (value_bfgs, "BFGS")
                                                                    ])
        utils.plot_contour(f, [-2, 2], [-1, 3], title=name, level=30, algorithm_paths=[(path_gd, "gradient descent"),
                                                                                       (path_new, "newton"),
                                                                                       (path_sr1, "SR1"),
                                                                                       (path_bfgs, "BFGS")
                                                                                       ])

    def testLinear(self):
        f = ex.linear_function
        name = "linear function "
        minimizer = unconstrained_min.LineSearchMinimization(f)
        minimizer.gradient_descent()
        path_gd, value_gd = minimizer.path, minimizer.values
        minimizer.newton()
        path_new, value_new = minimizer.path, minimizer.values
        minimizer.SR1()
        path_sr1, value_sr1 = minimizer.path, minimizer.values
        minimizer.BFGS()
        path_bfgs, value_bfgs = minimizer.path, minimizer.values
        utils.plot_value_iterations(name_of_function=name, methods=[(value_gd, "gradient descent"),
                                                                    (value_new, "newton"),
                                                                    (value_sr1, "SR1"),
                                                                    (value_bfgs, "BFGS")
                                                                    ])
        utils.plot_contour(f, [-650, 50], [-650, 50], title=name, level=50,
                           algorithm_paths=[(path_gd, "gradient descent"),
                                            (path_new, "newton"),
                                            (path_sr1, "SR1"),
                                            (path_bfgs, "BFGS")
                                            ])

    def testExponential(self):
        f = ex.exponential_func
        name = "exponential function "
        minimizer = unconstrained_min.LineSearchMinimization(f)
        minimizer.gradient_descent()
        path_gd, value_gd = minimizer.path, minimizer.values
        minimizer.newton()
        path_new, value_new = minimizer.path, minimizer.values
        minimizer.SR1()
        path_sr1, value_sr1 = minimizer.path, minimizer.values
        minimizer.BFGS()
        path_bfgs, value_bfgs = minimizer.path, minimizer.values
        utils.plot_value_iterations(name_of_function=name, methods=[(value_gd, "gradient descent"),
                                                                    (value_new, "newton"),
                                                                    (value_sr1, "SR1"),
                                                                    (value_bfgs, "BFGS")
                                                                    ])
        utils.plot_contour(f, [-2, 2], [-2, 2], title=name, level=10, algorithm_paths=[(path_gd, "gradient descent"),
                                                                                       (path_new, "newton"),
                                                                                       (path_sr1, "SR1"),
                                                                                       (path_bfgs, "BFGS")
                                                                                       ])


if __name__ == '__main__':
    unittest.main()
