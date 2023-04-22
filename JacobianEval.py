import sympy as sy


class JacobianEval:
    def __init__(self, funcs, vars):
        self.funcs = funcs
        self.vars = vars
        self.jac_m = sy.zeros(len(vars), len(vars))
        self.His_m = sy.zeros(len(vars), len(vars))

    # 获取雅克比矩阵
    def jacobi(self):
        self.jac_m = self.funcs.jacobian(self.vars)
        return self.jac_m

    # 获取海塞矩阵
    def hessia(self):
        for i, fi in enumerate(self.funcs):
            for j, r in enumerate(self.vars):
                for k, s in enumerate(self.vars):
                    self.His_m[k, j] = sy.diff(sy.diff(fi, r), s)
        return self.His_m


if __name__ == "__main__":

    def vdot1(x):
        return x
    # 符号化
    vdot, vdot2, y1, y2 = sy.symbols("vdot, vdot2, y1, y2")
    # 构建方程组
    vdot = vdot1(y2)
    vdot2 = 1000 * (1 - y1 ** 2) * y2 - y1
    # 封装成sympy的符号矩阵
    funcs = sy.Matrix([vdot, vdot2])
    args = sy.Matrix([y1, y2])
    # 实例化类并计算雅克比矩阵
    h = JacobianEval(funcs, args)
    jac = h.Ja()
    his = h.His()
    # 将sympy的符号转化为函数表达式
    f = sy.lambdify([y1, y2], his, 'numpy')  # 通过这段话转化为可以计算的函数表达式
    print(jac)
    print(his)
    print(f(1, 2))