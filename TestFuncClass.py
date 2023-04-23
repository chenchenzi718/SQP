import numpy as np


class TestFunc:
    def __init__(self, test_func_str="Rosenbrock"):
        self.test_func_str = test_func_str

    def test_func_val(self, input_val):
        # define a function named Rosenbrock
        if self.test_func_str == "Rosenbrock":
            return np.sum(100.0*(input_val[1]-input_val[0]**2.0)**2.0 + (1-input_val[0])**2.0)

    def test_func_constraint(self):
        # define the test_func's constraint
        if self.test_func_str == "Rosenbrock":
            cons_eq_1 = {'type': 'eq', 'fun': lambda x:  2 * x[0] + x[1] - 1}
            cons_ineq_1 = {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 1}
            cons_ineq_2 = {'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1] + 1, 'jac': lambda x: [-2 * x[0], -1]}
            cons_ineq_3 = {'type': 'ineq', 'fun': lambda x: -x[0]**2 + x[1] + 1, 'jac': lambda x: [-2 * x[0], 1]}
            cons = (cons_eq_1, cons_ineq_1, cons_ineq_2, cons_ineq_3)
            bounds = ((0, 1), (-0.5, 2.0))
            return cons, bounds

