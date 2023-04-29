import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MySQP import *


# func内为TestFunc类，actual_x输入minimize函数得到的结果，my_sqp_medium_x输入的为mysqp函数迭代过程中产生的x_k
# 如果actual_x 与 my_sqp_medium_x 均取None，则意味着我想要进行多次取初值作图的操作，因此会重新计算多个初值对应结果，无需传入这两个参数
def pic_my_sqp(func_class: TestFunc, actual_x=None, my_sqp_medium_x=None, test_func_name="Rosenbrock"):
    func = func_class.test_func_val
    dx = 0.01
    dy = 0.01
    if test_func_name == "Rosenbrock":
        x = np.arange(-2., 2., dx)
        y = np.arange(-2., 2., dy)
        X, Y = np.meshgrid(x, y)
    elif test_func_name == "test_1":
        x = np.arange(-3., 3., dx)
        y = np.arange(-3., 3., dy)
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

    if (my_sqp_medium_x is not None) and (actual_x is not None):
        # 一个初值进行绘制
        # 在图上画出最终收敛点
        plt.scatter(actual_x[0], actual_x[1], marker="*", c="y", s=100)

        # 在图上画出在sqp迭代过程中的点
        plt_x = []
        plt_y = []
        for _array in my_sqp_medium_x:
            plt_x.append(_array[0])
            plt_y.append(_array[1])
        plt.plot(plt_x, plt_y, "w--")
        plt.title("Primary value x0 iteration in method "+str(test_func_name))
        plt.show()
    else:
        # 五个初值进行绘制
        mat = np.array([[-1.5, -1.], [-1., 1.], [1.0, 0.5], [0.5, -1.5], [1.5, -1.]])

        sqp = MySQP(input_func=func_class.test_func_val, cons=func_class.cons, bounds=func_class.bounds)

        for primary_val in mat:
            # optimize结果
            res = minimize(func_class.test_func_val, primary_val, method='SLSQP',
                           bounds=func_class.bounds, constraints=func_class.cons)
            # 我的sqp结果
            res_mine, _ = sqp.my_sqp(x0=primary_val)
            mysqp_intermedium_result = sqp.mysqp_intermedium_result

            # 在图上画出最终收敛点
            plt.scatter(res.x[0], res.x[1], marker="*", c="y", s=100)

            # 在图上画出在sqp迭代过程中的点
            plt_x = []
            plt_y = []
            for _array in mysqp_intermedium_result:
                plt_x.append(_array[0])
                plt_y.append(_array[1])
            plt.plot(plt_x, plt_y, "w--")

        plt.title("Multiple Primary value x0 iteration in method "+str(test_func_name))
        plt.show()


if __name__ == '__main__':

    # test_func_name 可以取 "Rosenbrock" 或者 “test_1”
    test_func_name = "test_1"

    # 建立TestFunc类，根据输入的不同测试函数选择不同的约束条件
    test_func_class = TestFunc(test_func_str=test_func_name)
    cons, bounds = test_func_class.test_func_constraint()
    x0 = np.array([0.5, 0])
    test_func = test_func_class.test_func_val

    # 使用内置的minimize求出“精确解”
    # 我的函数与minimize函数对于Rosenbrock问题均有较好的结果，但是对于test_1函数则不是
    # 当取x0=[0.5, 0]时我写的函数与minimize函数都可以找到test_1的极小值[0.87403194 2.28824573]
    # 但是x0=[2., -1.5]时，minimize会失效（不知道为什么我的反而可以找到）
    res = minimize(test_func, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print(f"result of optimize.minimize function:{res.x}")

    # 用自己写的MySQP类求出自己解出的解
    sqp = MySQP(input_func=test_func_class.test_func_val, cons=cons, bounds=bounds)
    res_mine, lagrange_multiplier = sqp.my_sqp(x0=x0)
    print(f"result of my sqp function:{res_mine}")

    # 仅画出在x0条件下上述迭代的图示
    pic_my_sqp(test_func_class, res.x, sqp.mysqp_intermedium_result, test_func_name=test_func_name)

    # 可选操作，画出不同初值下的函数的结果
    multi_x0_res_flag = True
    if multi_x0_res_flag:
        pic_my_sqp(test_func_class, res.x, test_func_name=test_func_name)
