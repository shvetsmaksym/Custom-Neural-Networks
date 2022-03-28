import numpy as np
from DataStructures.Neurons import Neuron


class Layer:
    def __init__(self, n_neurons, prev_layer, input_shape):
        if prev_layer is not None:
            self.prev_layer = prev_layer
            self.prev_layer.next_layer = self
            self.neurons = [Neuron(layer=self, i=i + 1, input_neurons=self.prev_layer) for i
                            in range(n_neurons)]
        else:
            # Only for first layer
            self.neurons = [Neuron(layer=self, i=i + 1, input_shape=input_shape) for i
                            in range(n_neurons)]

        self.next_layer = None

    def activate_neurons(self, X_row=None):
        for neuron in self.neurons:
            neuron.activate(inputs=X_row)

    def calculate_error_signals(self, output=None):
        """output parameter is only for the last layer."""
        if output is not None:
            for neuron, out in zip(self.neurons, output):
                neuron.calculate_error_signal(out)
        else:
            for neuron in self.neurons:
                neuron.calculate_error_signal()

    def update_weights(self, lr):
        for neuron in self.neurons:
            neuron.update_weights(lr=lr)

    def __len__(self):
        return len(self.neurons)


class Input(Layer):
    def __init__(self, n_neurons, input_shape, prev_layer=None):
        super().__init__(n_neurons, prev_layer, input_shape)
        self.L = 1

    def update_weights_for_input(self, lr, inputs):
        for neuron in self.neurons:
            neuron.update_weights(lr=lr, inputs=inputs)


class Output(Layer):
    def __init__(self, n_neurons, prev_layer, input_shape=None):
        super().__init__(n_neurons, prev_layer, input_shape)
        self.L = -1  # Layer number
        self.error_function = self.calculate_error_for_binary if n_neurons == 1 \
            else self.calculate_error_for_multiclass

    def calculate_error_for_binary(self, Y, cl=False):
        """cl = False: return the error for one instance Y.
        cl = True: return True if classification is correct, return False if not."""
        if cl:
            return Y != round(self.neurons[0].F_net)
        else:
            return abs(Y - self.neurons[0].F_net)

    def calculate_error_for_multiclass(self, Y, cl=False):
        if cl:
            return np.where(Y == 1)[0][0] != np.argmax([neur.F_net for neur in self.neurons])
        else:
            return sum([abs(neur.F_net - y) for neur, y in zip(self.neurons, Y)])


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
