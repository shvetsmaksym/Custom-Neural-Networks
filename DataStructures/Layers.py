import numpy as np
from DataStructures.Neurons import Neuron
from DataStructures.ActivationFunctions import *


class Layer:
    def __init__(self, n_neurons, prev_layer, input_shape, activate=UnipolarSigmoid, discretization=False):
        if prev_layer is not None:
            self.prev_layer = prev_layer
            self.prev_layer.next_layer = self
            self.neurons = [Neuron(layer=self, i=i + 1, activation_func=activate) for i
                            in range(n_neurons)]
        else:
            # Only for first layer
            self.neurons = [Neuron(layer=self, i=i + 1, activation_func=activate, input_shape=input_shape) for i
                            in range(n_neurons)]

        self.next_layer = None

        self.discretize_fnets = discretization

    def activate_neurons(self, X_row=None):
        for neuron in self.neurons:
            neuron.activate(inputs=X_row)

        if self.discretize_fnets:
            self.discretization()

    def calculate_error_signals(self, output=None):
        """output is only given in the last layer."""
        if output is not None:
            for neuron, out in zip(self.neurons, output):
                neuron.calculate_error_signal(out)
        else:
            for neuron in self.neurons:
                neuron.calculate_error_signal()

    def update_weights(self, lr):
        for neuron in self.neurons:
            neuron.update_weights(lr=lr)

    def discretization(self, bits=4, r=(0, 1)):
        """Convert continuous value from [0, 1] range into discrete value.
        E.g. if bits == 1:
                [0, 0.5) -> 0
                [0.5, 1] -> 1
        """
        F_nets = np.array([neu.F_net for neu in self.neurons])
        range_ = r[1] - r[0]
        discrete_thresholds = np.array([range_ / i for i in range(2**bits, 0, -1)])
        discrete_fnets = np.digitize(F_nets, discrete_thresholds)

        for neu, d_fnet in zip(self.neurons, discrete_fnets):
            neu.F_net = d_fnet

    def __len__(self):
        return len(self.neurons)


class Input(Layer):
    def __init__(self, n_neurons, input_shape, prev_layer=None, activate=UnipolarSigmoid, discretization=False):
        super().__init__(n_neurons, prev_layer, input_shape, activate, discretization)
        self.L = 1

    def update_weights_for_input(self, lr, inputs):
        for neuron in self.neurons:
            neuron.update_weights(lr=lr, inputs=inputs)


class Output(Layer):
    def __init__(self, n_neurons, prev_layer, input_shape=None, activate=UnipolarSigmoid, discretization=False):
        super().__init__(n_neurons, prev_layer, input_shape, activate, discretization)
        self.L = -1  # Layer number

    def calculate_error(self, Y):
        return sum([abs(neur.F_net - y) for neur, y in zip(self.neurons, Y)])


class Hidden(Layer):
    def __init__(self, L, n_neurons, prev_layer, input_shape=None, activate=UnipolarSigmoid, discretization=False):
        super().__init__(n_neurons, prev_layer, input_shape, activate, discretization)
        self.L = L  # Layer number


if __name__ == "__main__":
    X_train = np.array([[0.15, 0.39, 0.94, 0.48],
                        [0.22, 0.38, 0.91, 0.39]])
    y_train = np.array([[0, 1],
                       [1, 0]])

    inp = Input(n_neurons=8, input_shape=4)
    hidden1 = Hidden(L=2, n_neurons=12, prev_layer=inp)
    output1 = Output(n_neurons=2, prev_layer=hidden1)

    # Feed forward
    inp.activate_neurons(X_row=X_train[0])
    hidden1.activate_neurons()
    output1.activate_neurons()

    # Calculate error signals backwards
    output1.calculate_error_signals(output=y_train[0])
    hidden1.calculate_error_signals()
    inp.calculate_error_signals()

    # Update weights
    inp.update_weights()
    hidden1.update_weights()
    output1.update_weights()

    # Feed forward (second row)
    inp.activate_neurons(X_row=X_train[1])
    hidden1.activate_neurons()
    output1.activate_neurons()

    # Calculate error signals backwards
    output1.calculate_error_signals(output=y_train[1])
    hidden1.calculate_error_signals()
    inp.calculate_error_signals()

    # Update weights
    inp.update_weights()
    hidden1.update_weights()
    output1.update_weights()

    # Calculate error

    print("Finish")
