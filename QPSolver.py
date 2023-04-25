from qpsolvers import Problem, solve_problem
import numpy as np

"""
   这一个类主要要解决的问题：
   min 1/2(xT)Px+(qT)x 
    subject to  Gx<=h
                Ax=b
                lb<=x<=ub
"""

class MyQPSolver:
    # 输入为W_k hessian阵，g_k 梯度向量，cons 不等式与等式约束，cons_func_jac为所有约束向量的梯度函数值，cons_func_val为约束向量值
    def __init__(self, W_k, g_k, cons, cons_func_jac, cons_func_val):
        self.W = W_k
        self.g = g_k
        self.cons = cons
        self.cons_func_jac = cons_func_jac
        self.cons_func_val = cons_func_val

        self.ineq_cons = []
        self.eq_cons = []
        self.divide_cons()
        self.eq_num = len(self.eq_cons)
        self.ineq_num = len(self.ineq_cons)

        self.P = self.W
        self.q = self.g
        self.G = None
        self.h = None
        self.A = None
        self.b = None
        self.construct_matrix()

    # 将cons 划分为等式与不等式条件
    def divide_cons(self):
        for cons in self.cons:
            if cons['type'] == 'eq':
                self.eq_cons.append(cons)
            else:
                self.ineq_cons.append(cons)

    # 建立整个矩阵系统
    def construct_matrix(self):
        self.A = self.cons_func_jac[:self.eq_num, :]
        self.b = -(self.cons_func_val[:self.eq_num])
        self.G = -(self.cons_func_jac[self.eq_num:, :])
        self.h = self.cons_func_val[self.eq_num:]

    # 解决qp问题的dual问题
    def qp_dual_solver(self):
        problem = Problem(self.P, self.q, self.G, self.h, self.A, self.b)
        solution = solve_problem(problem, solver="proxqp")

        return solution.x, np.append(solution.y, solution.z)


if __name__ == "__main__":
    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = M.T.dot(M)  # quick way to build a symmetric matrix
    q = np.array([3., 2., 3.]).dot(M).reshape((3,))
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    h = np.array([3., 2., -2.]).reshape((3,))
    A = np.array([1., 1., 1.])
    b = np.array([1.])
    lb = -0.6 * np.ones(3)
    ub = +0.7 * np.ones(3)

    cons = [
        {'type': 'eq', 'fun': lambda x:  x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] - x[2] + 3},
        {'type': 'ineq', 'fun': lambda x: -2 * x[0] - x[2] + 2},
        {'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + x[2] - 2},
        {'type': 'ineq', 'fun': lambda x: x[0] + 0.6},
        {'type': 'ineq', 'fun': lambda x: x[1] + 0.6},
        {'type': 'ineq', 'fun': lambda x: x[2] + 0.6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 0.7},
        {'type': 'ineq', 'fun': lambda x: -x[1] + 0.7},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.7},
    ]

    cons_func_jac = -np.array([[-1., -1., -1.], [1., 2., 1.], [2., 0., 1.], [-1., 2., -1.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.],
                   [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    cons_func_val = np.array([-1., 3., 2., -2., 0.6, 0.6, 0.6, 0.7, 0.7, 0.7])

    qp_sol = MyQPSolver(P, q, cons, cons_func_jac, cons_func_val)
    d, lamb = qp_sol.qp_dual_solver()
    print(f"Primal: d = {d}")
    print(f"Dual: lambda = {lamb}")


