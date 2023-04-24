import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MySQP import *


if __name__ == '__main__':

    test_func = TestFunc()
    cons, bounds = test_func.test_func_constraint()
    x0 = np.array([0.5, 0.0])
    rosen = test_func.test_func_val
    res = minimize(rosen, x0, method='SLSQP', bounds=bounds, constraints=cons)

    sqp = MySQP(input_func=test_func.test_func_val, cons=cons, bounds=bounds)
    y = sqp.g_k(x0)
    print(y)
    A = sqp.A_k(x0)
    print(A)
    c = sqp.c_k(x0)
    print(c)
