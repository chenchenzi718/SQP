import numpy as np

"""
    本文件完成了类TestFunc的编写，根据输入的test_func_str的不同决定不同的测试函数
    test_func_val函数计算对应的值
    同时也在本类中给出了对应于测试函数的约束条件设置test_func_constraint
"""


class TestFunc:
    def __init__(self, test_func_str="Rosenbrock"):
        self.test_func_str = test_func_str

        self.cons = None
        self.bounds = None

    def test_func_val(self, input_val):
        # define a function named Rosenbrock
        if self.test_func_str == "Rosenbrock":
            return 100.0*((input_val[1]-input_val[0]**2.0)**2.0) + (1-input_val[0])**2.0
        if self.test_func_str == "test_1":    # 返回一个f(x,y)=x^3-y^3+xy+2x^2
            return input_val[0]**3.0 - input_val[1]**3.0 + input_val[0] * input_val[1] + 2.0 * input_val[0]**2.0

    # 不等式约束均为大于等于的约束
    def test_func_constraint(self):
        # define the test_func's constraint
        if self.test_func_str == "Rosenbrock":
            cons_eq_1 = {'type': 'eq', 'fun': lambda x:  2 * x[0] + x[1] - 1}
            cons_ineq_1 = {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 1}
            cons_ineq_2 = {'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1] + 1, 'jac': lambda x: [-2 * x[0], -1]}
            cons_ineq_3 = {'type': 'ineq', 'fun': lambda x: -x[0]**2 + x[1] + 1, 'jac': lambda x: [-2 * x[0], 1]}
            cons = (cons_eq_1, cons_ineq_1, cons_ineq_2, cons_ineq_3)
            bounds = ((0, 1), (-0.5, 2.0))
            self.cons = cons
            self.bounds = bounds
            return cons, bounds
        if self.test_func_str == "test_1":
            cons_eq_1 = {'type': 'eq', 'fun': lambda x: x[0] * x[1] - 2}
            cons_ineq_1 = {'type': 'ineq', 'fun': lambda x:  6. - x[0]**2 - x[1]**2,
                           'jac': lambda x: [-2 * x[0], -2 * x[1]]}
            cons = (cons_eq_1, cons_ineq_1)
            self.cons = cons
            return cons, None
