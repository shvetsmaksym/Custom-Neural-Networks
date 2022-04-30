import numpy as np
import matplotlib.pyplot as plt
from DataStructures.ActivationFunctions import UnipolarSigmoid, BipolarSigmoid


class Neuron1:
    """Class corresponding to simple neuron, which is able to solve NOR, NAND"""

    def __init__(self, x, weights, thresh, outputs=(1, 0)):
        self.x = np.array(x)
        self.weights = np.array(weights)
        self.thresh = thresh
        self.outputs = outputs

    def calculate(self):
        net = np.dot(self.x, self.weights)
        if net >= self.thresh:
            return self.outputs[0]
        else:
            return self.outputs[1]


class Perceptron:
    """Class describing Perceptron"""

    def __init__(self, data: np.array, weights: np.array, thresh, outputs=(1, 0), lr=0.05, speed=8):
        if np.shape(data)[1] - 1 != len(weights):
            raise IndexError("Dimension Error. Number of X columns is not the same as the number of weights")
        else:
            self.data = data
            self.weights = weights

            self.thresh = thresh  # threshold for bipolar activation function
            self.outputs = outputs
            self.lr = lr

            self.graph = plt
            self.plot_update_speed = speed

    def perform_one_epoch(self):
        for data_row in self.data:
            f_net = self.activate_neuron(data_row=data_row)
            self.update_weights(f_net=f_net, data_row=data_row)
            self.draw_updated_line(data_row=data_row)

    def learn_perceptron(self, n_epochs=10):
        """Method for performing n_epochs of learning perceptron.
        If error == 0 then learning process is ended and weights are printed in a console."""

        for epoch in range(1, n_epochs + 1):
            self.perform_one_epoch()
            errors = self.calculate_error()
            print(f"===Epoch {epoch} \t Errors: {errors}")
            if errors == 0:
                print(f"Complete Learning Perceptron! Weights: {self.weights}")
                break

            self.shuffle_data()

    def calculate_error(self):
        """Return the number of incorrect classifications."""
        errors = 0
        for data_row in self.data:
            f_net = self.activate_neuron(data_row=data_row)
            r = data_row[-1] - f_net
            if r != 0:
                errors += 1
        return errors

    def activate_neuron(self, data_row):
        """Calculate the output of net function for a single row from X"""

        net = np.dot(data_row[:-1], self.weights)

        if net >= self.thresh:
            return self.outputs[0]
        else:
            return self.outputs[1]

    def update_weights(self, f_net, data_row):
        r = data_row[-1] - f_net
        delta_weights = self.lr * r * data_row[:-1]
        self.weights = self.weights + delta_weights

    def prepare_plot(self, x_lim=(-5, 5), y_lim=(-5, 5)):
        """Prepare plot and draw the points from training dataset."""

        self.graph.clf()
        self.graph.style.use('fivethirtyeight')
        self.graph.xlim(x_lim)
        self.graph.ylim(y_lim)
        self.graph.xlabel('x1', size=15)
        self.graph.ylabel('x2', size=15)

        # Plot dots with y = -1
        self.graph.plot(self.data[self.data[:, -1] == -1, 1], self.data[self.data[:, -1] == -1, 2], 'ro')
        # Plot dots with y = 1
        self.graph.plot(self.data[self.data[:, -1] == 1, 1], self.data[self.data[:, -1] == 1, 2], 'bo')

    def draw_updated_line(self, data_row, x_range=range(-6, 6)):
        """Draw the line based on updated weights.
        Also mark the point from training dataset corresponding to this update."""

        self.prepare_plot()

        x1 = np.array(x_range)
        x2 = -(self.weights[1] * x1 + self.weights[0]) / self.weights[2]
        self.graph.plot(x1, x2, color='black', linewidth=2)
        self.graph.plot(data_row[1], data_row[2], 'o', color='black', linewidth=2, markersize=20,
                        markerfacecolor='none')

        self.graph.text(data_row[1] + 0.3, data_row[2] + 0.3, f"[{data_row[1]};{data_row[2]}] Y={data_row[3]}")
        self.graph.title(" ".join(["w" + str(i) + "=" + str(round(x, 3)) + ";  " for i, x in enumerate(self.weights)]),
                         size=12)
        self.graph.pause(1 / self.plot_update_speed)

    def shuffle_data(self):
        np.random.shuffle(self.data)

    def show_plot(self):
        self.graph.show()


class Neuron:
    def __init__(self, layer, i, activation_func, input_shape=None, output_neurons=None):

        self.layer = layer      # warstwa
        self.i = i              # numer neuronu w danej warstwie

        self.input_shape = input_shape
        self.output_neurons = output_neurons
        self.F_net = 0
        self.delta = 0          # sygnał błędu
        self.activation_function = activation_func(lbd=1)

        if self.input_shape:
            # First Layer
            self.weights = (np.random.random(self.input_shape + 1) - 1/2) * 2
        else:
            # 2:L Layers
            self.weights = (np.random.random(len(self.layer.prev_layer.neurons) + 1) - 1/2) * 2

    def activate(self, inputs=None):
        """Activate neuron with activation function.
        Inputs can be given as a numpy array (Input Layer) or as neuron objects."""
        if type(inputs) is np.ndarray:
            # First layer
            inputs_with_bias = np.insert(inputs, 0, 1)
        else:
            # 2:L layers - inputs are neurons
            values = np.array([neu.F_net for neu in self.layer.prev_layer.neurons])
            inputs_with_bias = np.insert(values, 0, 1)

        net = np.dot(inputs_with_bias, self.weights)
        self.F_net = self.activation_function.func(net)

    def calculate_error_signal(self, out=None):
        if out is not None:
            # Last layer
            self.delta = out - self.F_net
        else:
            # Layers 1: L-1
            wgh = np.array([neuron.weights[self.i] for neuron in self.layer.next_layer.neurons])
            err_signals = np.array([neuron.delta for neuron in self.layer.next_layer.neurons])
            self.delta = np.dot(wgh, err_signals)

    def update_weights(self, lr=0.01, inputs=None):
        if self.layer.L != 1:
            inputs = np.array([neuron.F_net for neuron in self.layer.prev_layer.neurons])

        # Add bias = 1
        inputs = np.insert(inputs, 0, 1)

        # Calculate deltas for weights and update weights
        delta_weights = np.array([lr * self.delta * self.activation_function.derivative(self.F_net) * inp for inp in inputs])
        self.weights = self.weights + delta_weights




