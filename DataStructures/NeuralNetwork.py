from DataStructures.Layers import *
from random import random, randint
# from DataStructures.Neurons import Neuron


class NeuralNetwork:
    def __init__(self):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def add_input_layer(self, n, input_shape):
        self.input_layer = Input(n_neurons=n, input_shape=input_shape)

    def add_hidden_layer(self, L, n):
        if self.hidden_layers:
            hidden_layer = Hidden(L, n_neurons=n, prev_layer=self.hidden_layers[-1])
            self.hidden_layers.append(hidden_layer)

        elif self.input_layer:
            hidden_layer = Hidden(L, n_neurons=n, prev_layer=self.input_layer)
            self.hidden_layers.append(hidden_layer)

        else:
            print("There are no input layer yet.")

    def add_output_layer(self, n):
        if self.hidden_layers:
            self.output_layer = Output(n_neurons=n, prev_layer=self.hidden_layers[-1])
        elif self.input_layer:
            self.output_layer = Output(n_neurons=n, prev_layer=self.input_layer)
        else:
            print("There are no hidden nor input layers.")

    def feed_forward(self, xi):
        self.input_layer.activate_neurons(X_row=xi)
        if self.hidden_layers:
            for hl in self.hidden_layers:
                hl.activate_neurons()
        if self.output_layer:
            self.output_layer.activate_neurons()

    def propagate_errors(self, yi):
        if self.output_layer:
            self.output_layer.calculate_error_signals(output=yi)
        if self.hidden_layers:
            for hl in self.hidden_layers:
                hl.activate_neurons()
        if self.hidden_layers == [] and self.output_layer is None:
            self.input_layer.calculate_error_signals(output=yi)
        else:
            self.input_layer.calculate_error_signals()

    def update_weights(self, lr):
        for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]:
            layer.update_weights(lr=lr)

    def calculate_errors(self, validation_data):
        error = 0
        if validation_data:
            for xi, yi in zip(validation_data[0], validation_data[1]):
                self.feed_forward(xi)
                error += self.output_layer.calculate_error_for_one_sample(yi)
        return error

    def perform_one_epoch(self, train_x, train_y, lr):
        for xi, yi in zip(train_x, train_y):
            self.feed_forward(xi)
            self.propagate_errors(yi)
            self.update_weights(lr=lr)

    def fit(self, train_x, train_y, validation_data=None, lr=0.01, epochs=10):
        for epoch in range(epochs):
            self.perform_one_epoch(train_x, train_y, lr=lr)
            error = self.calculate_errors(validation_data=validation_data)
            print(f"Epoch {epoch}, error: {error}")


def convert_y_multiclass_to_binary(y, classes):
    y_out = np.array([[0 for c in range(classes)] for _ in y])
    for i, yi, in enumerate(y):
        y_out[i, yi-1] = 1

    return y_out


if __name__ == "__main__":
    X_train = np.array([[random() for i in range(4)] for _ in range(10)])
    y_train = np.array([1,0,0,1,0,1,0,1,0,1])
    y_train = convert_y_multiclass_to_binary(y=y_train, classes=2)

    X_test = np.array([[random() for i in range(4)] for _ in range(4)])
    y_test = np.array([1,0,0,1])
    y_test = convert_y_multiclass_to_binary(y=y_test, classes=2)

    model = NeuralNetwork()
    model.add_input_layer(n=8, input_shape=4)
    model.add_hidden_layer(L=2, n=16)
    model.add_output_layer(n=2)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

    print("Finish.")
