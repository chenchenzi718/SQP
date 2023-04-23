from JacobianEval import *


class MySQP:
    """
        默认输入的是二维的问题，input_func对应输入的优化函数，cons为约束，bounds对应着区间,epi对应着最终收敛结束的值，sigma为罚函数的参数
        x0为输入的初始值
    """
    def __init__(self, input_func, cons, bounds, epi, sigma, x0=np.array([0.5, 0.0])):
        self.func = input_func
        self.cons = cons
        self.bounds = bounds
        self.epi = epi
        self.sigma = sigma
        self.x0 = x0

        self.cons_with_bounds = []

    # 想要把两个变量的区间信息直接作为一个函数不等式约束
    def change_bounds_to_cons(self):
        x0_range = self.bounds[0]
        x1_range = self.bounds[1]
        if x0_range[0] is not None:
            self.cons_with_bounds.append({'type': 'ineq', 'fun': lambda x: x[0] - x0_range[0]})
        if x0_range[1] is not None:
            self.cons_with_bounds.append({'type': 'ineq', 'fun': lambda x: -x[0] + x0_range[1]})
        if x1_range[0] is not None:
            self.cons_with_bounds.append({'type': 'ineq', 'fun': lambda x: x[1] - x1_range[0]})
        if x1_range[1] is not None:
            self.cons_with_bounds.append({'type': 'ineq', 'fun': lambda x: -x[1] + x1_range[1]})

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
        for cons in self.cons_with_bounds:
            penalty += self.sigma * abs(c_(cons, x))

        return penalty

    @staticmethod
    def Jacobi(input_func, x):

