import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MySQP import *


# func内为待优化函数，actual_x输入minimize函数得到的结果，my_sqp_medium_x输入的为mysqp函数迭代过程中产生的x_k
def pic_my_sqp(func, actual_x, my_sqp_medium_x, bound=None):
    dx = 0.001
    dy = 0.001
    if bound is None:
        d_max = np.max(np.abs(actual_x))
        x = np.arange(-5.0 * d_max, 5.0 * d_max, dx)
        y = np.arange(-5.0 * d_max, 5.0 * d_max, dy)
        X, Y = np.meshgrid(x, y)
    else:
        x = np.arange(-2., 2., dx)
        y = np.arange(-2., 2., dy)
        X, Y = np.meshgrid(x, y)

    y1, y2 = sy.symbols("y1, y2")
    func_sym = func([y1, y2])
    func_val = sy.lambdify([y1, y2], func_sym, 'numpy')

    contour = plt.contour(X, Y, func_val(X, Y), 20)  # 生成等值线图
    plt.contourf(X, Y, func_val(X, Y), 20)
    # plt.clabel(contour, inline=1, fontsize=10)
    plt.colorbar()

    plt.scatter(actual_x[0], actual_x[1], marker="*")

    plt.show()


if __name__ == '__main__':

    test_func_name = "Rosenbrock"
    test_func = TestFunc(test_func_str=test_func_name)
    cons, bounds = test_func.test_func_constraint()
    x0 = np.array([0.5, 0.])
    rosen = test_func.test_func_val
    res = minimize(rosen, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(res.x)

    sqp = MySQP(input_func=test_func.test_func_val, cons=cons, bounds=bounds)
    res_mine, lagrange_multiplier = sqp.my_sqp()
    print(res_mine)

    pic_my_sqp(rosen, res.x, sqp.mysqp_intermedium_result, bound=bounds)
