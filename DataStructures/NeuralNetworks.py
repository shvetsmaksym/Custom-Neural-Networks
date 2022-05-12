from DataStructures.Layers import *
from random import random, sample
from DataStructures.Preprocessing import one_hot_encoding
from DataStructures.Callbacks import Metrics, EarlyStopping, is_correct_class, print_progress_bar
from DataStructures.ActivationFunctions import Gaussian


class NeuralNetwork:
    def __init__(self, classification_problem=True):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None

        if classification_problem:
            self.basic_metrics_history = {"errors": [], "correct_classifications": []}
        else:
            self.basic_metrics_history = {"errors": []}

        self.metrics = Metrics(neural_net=self, metrics=['mse'])
        self.metric_history = {}

        self.early_stopping = None

        self.classification_problem = classification_problem

    def add_metrics(self, metrics=None):
        if metrics is not None:
            try:
                self.metrics = Metrics(neural_net=self, metrics=metrics)
            except KeyError as e:
                print(e, "\nSome of the given metrics is invalid.")

    def add_early_stopping(self, patience, monitor='mse', min_delta=0.01):
        self.early_stopping = EarlyStopping(self, patience=patience, monitor=monitor, min_delta=min_delta)

    def add_input_layer(self, n, input_shape, discretization=False):
        self.input_layer = Input(n_neurons=n, input_shape=input_shape, discretization=discretization)

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
                hl.calculate_error_signals()
        if self.hidden_layers == [] and self.output_layer is None:
            self.input_layer.calculate_error_signals(output=yi)
        else:
            self.input_layer.calculate_error_signals()

    def update_weights(self, lr, xi):
        self.input_layer.update_weights_for_input(lr=lr, inputs=xi)
        for layer in self.hidden_layers + [self.output_layer]:
            layer.update_weights(lr=lr)

    def calculate_errors(self, validation_data, classification_problem=True):
        errors = np.zeros(len(validation_data[0]))
        res = [np.zeros(len(self.output_layer.neurons)) for _ in range(len(validation_data[1]))]

        if validation_data:
            for i, xi in enumerate(validation_data[0]):
                yi = np.array([validation_data[1][i]]).flatten()

                self.feed_forward(xi)
                res[i] = np.array([neuron.F_net for neuron in self.output_layer.neurons])
                errors[i] = np.sum(abs(yi - res[i]))

            if self.classification_problem:
                correct_classifications = np.zeros(len(validation_data[0]))
                for i in range(len(validation_data[1])):
                    yi = np.array([validation_data[1][i]]).flatten()
                    correct_classifications[i] = 1 if is_correct_class(F_nets=res[i], Y=yi) else 0
                return errors, correct_classifications
            else:
                return errors, None

    def calculate_metrics(self, epoch, freq=1):
        if epoch % freq == 0:
            print("{0:10}".format(f"Epoch {epoch + 1}"), end='\t')
            verbose = 1
        else:
            verbose = 0
        self.metrics.calculate_metrics(verbose=verbose)

    def perform_one_epoch(self, epoch, train_x, train_y, lr):
        count = 0
        # train on sample
        # train_samples = sample([x for x in range(len(train_x))], 2048)
        # for xi, yi in zip(train_x[train_samples], train_y[train_samples]):
        for xi, yi in zip(train_x, train_y):
            self.feed_forward(xi)
            self.propagate_errors(yi)
            self.update_weights(xi=xi, lr=lr)
            count += 1
            if count % 16 == 0:
                print_progress_bar(text=f"Epoch {epoch + 1}", a=count, b=len(train_x))

    def fit(self, train_x, train_y, validation_data=None, lr=0.01, epochs=10):
        for epoch in range(epochs):
            # Shuffle train data
            if len(train_y.shape) > 1:
                train_data = np.concatenate((train_x, train_y), axis=1)
            else:
                # For only binary problems
                train_data = np.column_stack((train_x, train_y))
            np.random.shuffle(train_data)
            train_x_shuffled, train_y_shuffled = train_data[:, :len(train_x[-1, :])], train_data[:, len(train_x[-1, :]):]

            # Perform epoch
            self.perform_one_epoch(epoch, train_x_shuffled, train_y_shuffled, lr=lr)

            # Calculate errors
            err, corr_cl = self.calculate_errors(validation_data=validation_data, classification_problem=False)
            self.basic_metrics_history["errors"].append(err)
            if corr_cl is not None:
                self.basic_metrics_history["correct_classifications"].append(corr_cl)
            self.calculate_metrics(epoch)

            # Check for early stopping
            if self.early_stopping is not None:
                if self.early_stopping.check_es():
                    break


class RBF:
    def __init__(self, K, Beta=1):
        self.X = None
        self.Y = None
        self.K = K  # number of neurons in nonlinear layer
        self.k_samples = None
        self.weights = np.zeros(self.K)
        self.a_func = Gaussian(Beta=Beta)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.k_samples = np.random.choice(self.X, size=self.K, replace=False)

        R = self.calculate_R_matrix()
        self.weights = np.dot(np.linalg.pinv(R), Y)

    def calculate_R_matrix(self):
        """Calculate outputs from radial layer"""
        R = np.zeros((len(self.X), self.K))
        for i, x in enumerate(self.X):
            for j, c in enumerate(self.k_samples):
                R[i, j] = self.a_func.func(x, c)
        return R

    def predict(self, x_):
        # x_ - array of inputs
        R = self.calculate_R_matrix()
        y = np.dot(R, self.weights)
        return y


if __name__ == "__main__":
    train_x = np.linspace(-1, 1, 50)
    train_y = np.sin((train_x + 0.5)**3) + np.random.randn(50) / 10
    print(train_y)