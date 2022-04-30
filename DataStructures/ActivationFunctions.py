from random import random
import numpy as np
import matplotlib.pyplot as plt


class ActivationFunction:

    def __init__(self, name, lbd=1):
        self.lbd = lbd
        self.name = name

    def func(self, x):
        pass

    def derivative(self, x):
        pass

    def plot(self, x1, x2, flag=0):
        """Plots function if flag == 0.
        Plots derivative of this function if flag == 1.
        """
        x = np.linspace(x1, x2, 100)

        if flag == 0:
            y = np.array([self.func(x_) for x_ in x])
        else:
            y = np.array([self.derivative(x_) for x_ in x])

        plt.plot(x, y, color=(random() * 0.9, random() * 0.8, random() * 0.8))


class UnipolarSigmoid(ActivationFunction):
    def __init__(self, name="Unipolar Sigmoid", lbd=1):
        super().__init__(name, lbd)

    def func(self, x):
        try:
            return 1 / (1 + np.exp(-self.lbd * x))
        except OverflowError as e:
            return 1 / (1 + np.exp(-self.lbd * x / 10))

    def derivative(self, x):
        return self.lbd * self.func(x) * (1 - self.func(x))


class BipolarSigmoid(ActivationFunction):
    def __init__(self, name="Bipolar Sigmoid", lbd=1):
        super().__init__(name, lbd)

    def func(self, x):
        return 2 / (1 + np.exp(-self.lbd * x)) - 1

    def derivative(self, x):
        return self.lbd * 1/2 * (1 - self.func(x) ** 2)


if __name__ == "__main__":
    x1, x2 = -2, 2
    lambdas = [0.2, 0.5, 1, 2, 4, 5]
    for l in lambdas:
        us = UnipolarSigmoid(lbd=l)
        us.plot(x1, x2)
    plt.legend([f"lbd:{l}" for l in lambdas])
    plt.show()