from JacobianEval import *
from QPSolver import *
from scipy.optimize import linesearch


class MySQP:
    """
        默认输入的是二维的问题，input_func对应输入的优化函数，cons为约束，bounds对应着区间,
        rou对应着罚函数一维搜索的区间范围，epi对应着最终收敛结束的值，sigma为罚函数的参数
        x0为输入的初始值
    """
    def __init__(self, input_func, cons, bounds, rou=1, epi=1e-5, sigma=100, x0=np.array([0.5, 0.0])):
        self.func = input_func
        self.cons = cons
        self.bounds = bounds
        self.epi = epi
        self.sigma = sigma
        self.x0 = x0
        self.rou = rou

        # 将bounds转化为cons存储为列表
        self.bounds_to_cons = []
        self.__change_bounds_to_cons()
        self.cons_with_bounds = list(self.cons) + self.bounds_to_cons

        self.test_func_jac = None
        self.test_func_hes = None
        self.cons_func_jac = []
        self.cons_func_hes = []
        self.__build_sy_jacobi_hessian()

    # 给出需要计算的函数以及约束函数的jacobi的符号计算结果
    def __build_sy_jacobi_hessian(self):
        y1, y2 = sy.symbols("y1, y2")
        test_func = self.func([y1, y2])

        # 封装成sympy的符号矩阵
        funcs = sy.Matrix([test_func])
        args = sy.Matrix([y1, y2])

        # 实例化类并计算雅克比矩阵
        h = JacobianEval(funcs, args)
        self.test_func_jac = h.jacobi()
        self.test_func_hes = h.hessian()

        for cons in self.cons_with_bounds:
            funcs = sy.Matrix([cons['fun']([y1, y2])])
            h = JacobianEval(funcs, args)
            self.cons_func_jac.append(h.jacobi())
            self.cons_func_hes.append(h.hessian())

    # 想要把两个变量的区间信息直接作为一个函数不等式约束
    def __change_bounds_to_cons(self):
        x0_range = self.bounds[0]
        x1_range = self.bounds[1]
        if x0_range[0] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: x[0] - x0_range[0]})
        if x0_range[1] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: -x[0] + x0_range[1]})
        if x1_range[0] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: x[1] - x1_range[0]})
        if x1_range[1] is not None:
            self.bounds_to_cons.append({'type': 'ineq', 'fun': lambda x: -x[1] + x1_range[1]})

    # 创建一个罚函数
    def penalty_function(self, x):
        def c_(constrain, y):
            if constrain['type'] == 'eq':
                return constrain['fun'](y)
            else:
                return min(0., constrain['fun'](y))

        penalty = self.func(x)
        for cons in self.cons:
            penalty += self.sigma * abs(c_(cons, x))
        for cons in self.bounds_to_cons:
            penalty += self.sigma * abs(c_(cons, x))

        return penalty

    # sqp中的计算jacobi函数，sym_jacobi为符号函数, 返回的为一个行向量
    def sqp_jacobi(self, sym_jacobi, x):
        y1, y2 = sy.symbols("y1, y2")
        f_jac = sy.lambdify([y1, y2], sym_jacobi, 'numpy')  # 通过这段话转化为可以计算的函数表达式
        return f_jac(x[0], x[1])

    # sqp中的计算hessian函数，sym_hessian为符号函数
    def sqp_hessian(self, sym_hessian, x):
        y1, y2 = sy.symbols("y1, y2")
        f_his = sy.lambdify([y1, y2], sym_hessian, 'numpy')
        return f_his(x[0], x[1])

    # 计算Wk(x,\lambda)=Hf(x)-sum(\lambda_i * Hc_i(x)), H表示计算Hessian，Wk为一个n*n矩阵，lambda为一个m*1的矩阵
    def Wk_lagrange_hessian(self, x, _lambda):
        W = self.sqp_hessian(self.test_func_hes, x)
        for i, cons in enumerate(self.cons_with_bounds):
            W -= _lambda[i] * self.sqp_hessian(self.cons_func_hes[i], x)
        return W

    # 计算梯度向量gk, 为一个n*1的矩阵
    def g_k(self, x):
        g = self.sqp_jacobi(self.test_func_jac, x)
        return g.T

    # 计算约束条件向量的梯度矩阵A,为m*n矩阵
    def A_k(self, x):
        A = []
        for cons_func_jac in self.cons_func_jac:
            g = self.sqp_jacobi(cons_func_jac, x)
            A.append(g[0])
        return np.array(A)

    # 计算约束条件向量c，大小为m*1
    def c_k(self, x):
        c = []
        for cons in self.cons_with_bounds:
            c.append([cons['fun'](x)])
        return np.array(c)

    # 计算原问题lagrange函数的数值
    def lagrange_func_val(self, x, _lambda):
        val = self.func(x)
        c = self.c_k(x)
        return val - (_lambda.reshape(1, -1)) @ c

    # 计算原问题lagrange函数的jacobi矩阵数值
    def lagrange_jacobi(self, x, _lambda):
        j = self.sqp_jacobi(self.test_func_jac, x).T
        for i, cons in enumerate(self.cons_with_bounds):
            j -= _lambda[i] * self.sqp_jacobi(self.cons_func_jac[i], x).T
        return j

    # 如果W_k不是正定的，去做正定矫正，mode可以为simple,其他的暂时未考虑
    def Wk_ortho_correct(self, W, mode="simple"):
        if mode == "simple":
            """
            此时做的是加上最小的负的特征值的绝对值方式调整，如果第一次迭代的hessian就是一个负定，那么使用BFGS也没用，因此第一次不
            是正定时要强硬的变成正定阵

            """
            eigenvalue, featurevector = np.linalg.eig(W)
            min_eigenvalue = np.min(eigenvalue)
            if min_eigenvalue <= 0:
                W += (abs(min_eigenvalue) + 1e-3) * np.eye(len(eigenvalue))
                return W
            else:
                return W

    # 一维搜索算法，要确定罚函数一维搜索的系数以及“最小”结果
    def my_linesearch(self, x_k, d_k):
        # 因为这个罚函数是不可导的，这里我们采用0.618法进行计算最小值
        # alpha取值范围为[0,rou]
        r = 0.618
        a = 0
        alpha = 1.
        b = self.rou
        x1 = b - r * (b - a)
        x2 = a + r * (b - a)

        epoch = 200
        for i in range(epoch):
            if abs(x1 - x2) < self.epi:
                alpha = (x1 + x2)/2.
                break

            f1 = self.penalty_function(x_k + x1 * d_k)
            f2 = self.penalty_function(x_k + x2 * d_k)

            if f1 > f2:
                a = x1
                x1 = x2
                x2 = a + r * (b - a)
            else:
                b = x2
                x2 = x1
                x1 = a + (1. - r) * (b - a)
        return alpha

    # sqp算法，得到函数的极小值，这里的W直接取的lagrange函数的hessian矩阵
    def my_sqp(self):
        k = 0
        x_k = self.x0
        lambda_k = np.zeros(len(self.cons_with_bounds))

        # 规定循环次数
        epoch = 200
        for k in range(epoch):
            W_k = self.Wk_lagrange_hessian(x_k, lambda_k)
            g_k = self.g_k(x_k)
            A_k = self.A_k(x_k)
            c_k = self.c_k(x_k)

            # 进行二次规划求解
            W_k = self.Wk_ortho_correct(W_k)   # 做一次正定判定与矫正
            qp_sol = MyQPSolver(W_k, g_k, cons=self.cons_with_bounds, cons_func_jac=A_k, cons_func_val=c_k)
            d_k, lagrange_multiplier_k = qp_sol.qp_dual_solver()

            if np.linalg.norm(d_k, ord=2) <= self.epi:
                break

            alpha = self.my_linesearch(x_k, d_k)
            x_k += alpha * d_k
            lambda_k += alpha * (lagrange_multiplier_k - lambda_k)

        # 输出
        return x_k, lambda_k

