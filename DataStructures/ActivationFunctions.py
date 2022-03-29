from math import exp


class UnipolarSigmoid:
    def __init__(self, lbd=2):
        self.lbd = lbd

    def func(self, x):
        return 1 / (1 + exp(-self.lbd * x))

    def derivative(self, x):
        return self.lbd * self.func(x) * (1 - self.func(x))


class BipolarSigmoid:
    def __init__(self, lbd=2):
        self.lbd = lbd

    def func(self, x):
        return 2 / (1 + exp(-self.lbd * x)) - 1

    def derivative(self, x):
        return self.lbd * 1/2 * (1 - self.func(x) ** 2)
