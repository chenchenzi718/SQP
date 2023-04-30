import sympy as sy
from TestFuncClass import *

"""
    本文件完成了使用符号计算直接求解一个矩阵的jacobian矩阵与hessian矩阵的过程
    并给出了将符号矩阵转化为数值的方法，在n_jacobi与n_hessian中呈现
"""

class JacobianEval:
    def __init__(self, funcs, vars):
        self.funcs = funcs
        self.vars = vars
        self.jac_m = sy.zeros(len(vars), len(vars))
        self.His_m = sy.zeros(len(vars), len(vars))

        self.jac_state = False
        self.hes_state = False

    # 获取雅克比矩阵
    def jacobi(self):
        self.jac_m = self.funcs.jacobian(self.vars)
        self.jac_state = True
        return self.jac_m

    # 返回numerical jacobi
    def n_jacobi(self, x):
        if not self.jac_m:
            self.jacobi()
        return (sy.lambdify(self.vars, self.jac_m, 'numpy'))(x[0], x[1])

    # 获取海塞矩阵
    def hessian(self):
        self.His_m = sy.hessian(self.funcs, self.vars)
        self.hes_state = True
        return self.His_m

    # 返回numerical hessian
    def n_hessian(self, x):
        if not self.hes_state:
            self.hessian()
        return (sy.lambdify(self.vars, self.His_m, 'numpy'))(x[0], x[1])


if __name__ == "__main__":

    test_flag = 0
    test_Rosenbrock = 1

    if test_flag:
        # 符号化
        test_func, y1, y2 = sy.symbols("test_func, y1, y2")

        # 构建方程组
        test_func = 1000 * (1 - y1 ** 2) * y2 - y1
        # 封装成sympy的符号矩阵
        funcs = sy.Matrix([test_func])
        args = sy.Matrix([y1, y2])

        # 实例化类并计算雅克比矩阵
        h = JacobianEval(funcs, args)
        jac = h.jacobi()
        his = h.hessian()

        # 将sympy的符号转化为函数表达式
        f_jac = sy.lambdify([y1, y2], jac, 'numpy')  # 通过这段话转化为可以计算的函数表达式
        f_his = sy.lambdify([y1, y2], his, 'numpy')
        print(f"the jacobian matrix is {jac}")
        print(f"the hessian matrix is {his}")
        print(f_jac(1, 2))
        print(h.n_jacobi(np.array([1, 2])))
        print(f_his(1, 2))
        print(h.n_hessian(np.array([1, 2])))

    if test_Rosenbrock:
        y1, y2 = sy.symbols("y1, y2")
        test_func = TestFunc().test_func_val([y1, y2])

        # 封装成sympy的符号矩阵
        funcs = sy.Matrix([test_func])
        args = sy.Matrix([y1, y2])

        # 实例化类并计算雅克比矩阵
        h = JacobianEval(funcs, args)
        jac = h.jacobi()
        his = h.hessian()

        # 将sympy的符号转化为函数表达式
        f_jac = sy.lambdify([y1, y2], jac, 'numpy')  # 通过这段话转化为可以计算的函数表达式
        f_his = sy.lambdify([y1, y2], his, 'numpy')
        print(f"the jacobian matrix is {jac}")
        print(f"the hessian matrix is {his}")
        print(f_jac(0.41494475, 0.1701105))
        print(f_jac(0.5, 0))
        print(f_his(0.41494475, 0.1701105))
