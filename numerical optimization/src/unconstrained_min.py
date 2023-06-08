import numpy as np


def SR1_B(b, prev_x, new_x, prev_g, new_g):
    s_k = new_x - prev_x
    y_k = new_g - prev_g
    y_minus_bs = (y_k - np.dot(b, s_k))
    if np.dot(y_minus_bs, s_k) == 0:
        return b
    b = b + np.outer(y_minus_bs, y_minus_bs.T) / (np.dot(y_minus_bs, s_k))
    return b


def BFGS_B(b, prev_x, new_x, prev_g, new_g):
    s_k = new_x - prev_x
    y_k = new_g - prev_g
    bs = np.dot(b, s_k)
    if (np.dot(y_k, s_k)) == 0 or (s_k.T.dot(b.dot(s_k))) == 0:
        return b
    b = b - ((np.outer(bs, bs)) / (s_k.T.dot(b.dot(s_k)))) + ((np.outer(y_k, y_k)) / (np.dot(y_k, s_k)))
    return b


class LineSearchMinimization:
    def __init__(self, f, x0=np.array([1, 1]).T, obj_tol=10 ** -12, param_tol=10 ** -8, max_iter=100):
        self.path = []
        self.values = []
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter

    def gradient_descent(self,wolfe=True):
        prev_x = self.x0
        prev_value, prev_gradient, _ = self.f(prev_x)
        self.values = [prev_value]
        self.path = [[prev_x[0]], [prev_x[1]]]
        print("gradient descent")
        print(f"iteration= 0,location=  {prev_x}, value =  {prev_value}")
        for i in range(self.max_iter):
            direction = -prev_gradient
            alpha = self.backtracking_with_wolfe(prev_x, prev_value, prev_gradient, direction) if wolfe else 1
            new_x = prev_x + (alpha * direction)
            new_value, new_gradient, _ = self.f(new_x)
            self.values.append(new_value)
            self.path[0].append(new_x[0])
            self.path[1].append(new_x[1])
            print(f"iteration= {i + 1},location=  {new_x}, value =  {new_value}")
            if np.linalg.norm(new_x - prev_x) < self.param_tol or np.abs(new_value - prev_value) < self.obj_tol:
                return new_x, new_value, True
            prev_x, prev_value, prev_gradient = new_x, new_value, new_gradient
        return prev_x, prev_value, False

    def newton(self,wolfe=True):
        prev_x = self.x0
        prev_value, prev_gradient, prev_hessian = self.f(prev_x, True)
        self.values = [prev_value]
        self.path = [[prev_x[0]], [prev_x[1]]]
        print("newton")
        print(f"iteration= 0,location=  {prev_x}, value =  {prev_value}")
        for i in range(self.max_iter):
            try:
                direction = -np.linalg.solve(prev_hessian, prev_gradient)
            except:
                return prev_x, prev_value, False
            alpha = self.backtracking_with_wolfe(prev_x, prev_value, prev_gradient, direction) if wolfe else 1
            new_x = prev_x + (alpha * direction)
            new_value, new_gradient, new_hessian = self.f(new_x, True)
            self.values.append(new_value)
            self.path[0].append(new_x[0])
            self.path[1].append(new_x[1])
            print(f"iteration= {i + 1},location=  {new_x}, value =  {new_value}")
            if np.linalg.norm(new_x - prev_x) < self.param_tol or np.abs(new_value - prev_value) < self.obj_tol:
                return new_x, new_value, True
            prev_x, prev_value, prev_gradient, prev_hessian = new_x, new_value, new_gradient, new_hessian
        return prev_x, prev_value, False

    def BFGS(self,wolfe=True):
        prev_x = self.x0
        prev_value, prev_gradient, b = self.f(prev_x, True)
        self.values = [prev_value]
        self.path = [[prev_x[0]], [prev_x[1]]]
        print("BFGS")
        print(f"iteration= 0,location=  {prev_x}, value =  {prev_value}")
        for i in range(self.max_iter):
            try:
                direction = -np.linalg.solve(b, prev_gradient)
            except:
                return prev_x, prev_value, False
            alpha = self.backtracking_with_wolfe(prev_x, prev_value, prev_gradient, direction) if wolfe else 1
            new_x = prev_x + (alpha * direction)
            new_value, new_gradient, _ = self.f(new_x)
            b = BFGS_B(b, prev_x, new_x, prev_gradient, new_gradient)
            self.values.append(new_value)
            self.path[0].append(new_x[0])
            self.path[1].append(new_x[1])
            print(f"iteration= {i + 1},location=  {new_x}, value =  {new_value}")
            if np.linalg.norm(new_x - prev_x) < self.param_tol or np.abs(new_value - prev_value) < self.obj_tol:
                return new_x, new_value, True
            prev_x, prev_value, prev_gradient = new_x, new_value, new_gradient
        return prev_x, prev_value, False

    def SR1(self,wolfe=True):
        prev_x = self.x0
        prev_value, prev_gradient, b = self.f(prev_x, True)
        self.values = [prev_value]
        self.path = [[prev_x[0]], [prev_x[1]]]
        print("SR1")
        print(f"iteration= 0,location=  {prev_x}, value =  {prev_value}")
        for i in range(self.max_iter):
            try:
                direction = -np.linalg.solve(b, prev_gradient)
            except:
                return prev_x, prev_value, False
            alpha = self.backtracking_with_wolfe(prev_x, prev_value, prev_gradient, direction) if wolfe else 1
            new_x = prev_x + (alpha * direction)
            new_value, new_gradient, _ = self.f(new_x)
            b = SR1_B(b, prev_x, new_x, prev_gradient, new_gradient)
            self.values.append(new_value)
            self.path[0].append(new_x[0])
            self.path[1].append(new_x[1])
            print(f"iteration= {i + 1},location=  {new_x}, value =  {new_value}")
            if np.linalg.norm(new_x - prev_x) < self.param_tol or np.abs(new_value - prev_value) < self.obj_tol:
                return new_x, new_value, True
            prev_x, prev_value, prev_gradient = new_x, new_value, new_gradient
        return prev_x, prev_value, False

    def backtracking_with_wolfe(self, prev_x, prev_value, gradient, direction, wolfe=0.01, backtracking=0.5):
        a = 1  # Initial step length
        while True:
            new_value, _, _ = self.f(prev_x + a * direction)
            if new_value <= prev_value + wolfe * a * np.dot(gradient, direction):
                break
            a *= backtracking

        return a