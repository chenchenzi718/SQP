import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MySQP import *


if __name__ == '__main__':
    test_func = TestFunc()
    cons, bounds = test_func.test_func_constraint()
    x0 = np.array([0.5, 0.0])
    rosen = test_func.test_func_val
    res = minimize(rosen, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(res.x)
    print(res.jac)
    print(res.njev)
    print("you have done this job!")
