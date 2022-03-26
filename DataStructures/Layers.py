import numpy as np
from DataStructures.Neurons import Neuron


class Layer:
    def __init__(self, n_neurons, prev_layer, input_shape):
        if prev_layer is not None:
            self.prev_layer = prev_layer
            self.prev_layer.next_layer = self
            self.neurons = [Neuron(layer=self, i=i + 1, input_neurons=self.prev_layer, activation_func='US') for i
                            in range(n_neurons)]
        else:
            # Only for first layer
            self.neurons = [Neuron(layer=self, i=i + 1, input_shape=input_shape, activation_func='US') for i
                            in range(n_neurons)]

        self.next_layer = None

    def activate_neurons(self, X_row=None):
        for neuron in self.neurons:
            neuron.activate(inputs=X_row)

    def calculate_error_signals(self, output=None):
        """output parameter corresponds to outputs in the last layer."""
        for neuron in self.neurons:
            neuron.calculate_error_signal(output)

    def update_weights(self, lr):
        for neuron in self.neurons:
            neuron.update_weights(lr=lr)

    def __len__(self):
        return len(self.neurons)


class Input(Layer):
    def __init__(self, n_neurons, input_shape, prev_layer=None):
        super().__init__(n_neurons, prev_layer, input_shape)
        self.L = 1


class Output(Layer):
    def __init__(self, n_neurons, prev_layer, input_shape=None):
        super().__init__(n_neurons, prev_layer, input_shape)
        self.L = -1  # Layer number

    def calculate_error_for_one_sample(self, Y):
        return sum([neur.F_net - y for neur, y in zip(self.neurons, Y)])


class Hidden(Layer):
    def __init__(self, L, n_neurons, prev_layer, input_shape=None):
        super().__init__(n_neurons, prev_layer, input_shape)
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
