import numpy as np
from DataStructures.Neurons import Neuron


class Input:
    def __init__(self, input_shape, n_neurons):
        self.L = 1  # Layer number
        self.input_shape = input_shape
        self.neurons = [Neuron(self.L, layer=self, i=i + 1, input_shape=self.input_shape, activation_func='US') for i in
                        range(n_neurons)]

        self.next_layer = None

    def activate_neurons(self, X_row):
        for neuron in self.neurons:
            neuron.activate(inputs=X_row)

    def calculate_error_signals(self):
        for neuron in self.neurons:
            neuron.calculate_error_signal()

    def update_weights(self):
        for neuron in self.neurons:
            neuron.update_weights()

    def __len__(self):
        return len(self.neurons)


class Output:
    def __init__(self, n_neurons, prev_layer):
        self.L = -1  # Layer number
        self.prev_layer = prev_layer
        self.prev_layer.next_layer = self
        self.neurons = [Neuron(self.L, layer=self, i=i + 1, input_neurons=self.prev_layer, activation_func='US') for i
                        in range(n_neurons)]

        self.next_layer = None

    def activate_neurons(self):
        for neuron in self.neurons:
            neuron.activate()

    def calculate_error_signals(self, output: np.ndarray):
        for neuron, out in zip(self.neurons, output):
            neuron.calculate_error_signal(out)

    def update_weights(self):
        for neuron in self.neurons:
            neuron.update_weights()

    def __len__(self):
        return len(self.neurons)


class Hidden:
    def __init__(self, L, n_neurons, prev_layer):
        self.L = L  # Layer number
        self.prev_layer = prev_layer
        self.prev_layer.next_layer = self
        self.neurons = [Neuron(self.L, layer=self, i=i + 1, input_neurons=self.prev_layer, activation_func='US') for i
                        in range(n_neurons)]

        self.next_layer = None

    def activate_neurons(self):
        for neuron in self.neurons:
            neuron.activate()

    def calculate_error_signals(self):
        for neuron in self.neurons:
            neuron.calculate_error_signal()

    def update_weights(self):
        for neuron in self.neurons:
            neuron.update_weights()

    def __len__(self):
        return len(self.neurons)


if __name__ == "__main__":
    X_train = np.array([[0.15, 0.39, 0.94, 0.48],
                        [0.22, 0.38, 0.91, 0.39]])
    y_train = np.array([[0, 1],
                       [1, 0]])

    inp = Input(input_shape=4, n_neurons=8)
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


    print("Finish")
