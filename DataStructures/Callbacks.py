import numpy as np
import math


class Metrics:
    def __init__(self, neural_net, metrics: list):
        self.neural_net = neural_net

        self.all_metrics_functions = {
            'acc': self.accuracy,
            'mse': self.MSE
        }

        self.metrics = {m: self.all_metrics_functions[m] for m in metrics}
        self.neural_net.metric_history = {m: [] for m in metrics}

    def calculate_metrics(self, verbose=0):
        for func in self.metrics.values():
            func(verbose)
        print("\n")

    def accuracy(self, verbose=0):
        res = np.average(self.neural_net.basic_metrics_history['correct_classifications'][-1])
        self.neural_net.metric_history['acc'].append(res)
        if verbose:
            print("{0:25}".format(f"Accuracy: {round(res, 4)}"), end="\t")

    def MSE(self, verbose=0):
        res = 1/2 * (np.average(self.neural_net.basic_metrics_history['errors'][-1]) ** 2)
        self.neural_net.metric_history['mse'].append(res)
        if verbose:
            print("{0:25}".format(f"MSE: {round(res,4)}"), end="\t")


class EarlyStopping:
    def __init__(self, neural_net, patience, monitor, min_delta):
        self.neural_net = neural_net
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta

        modes_for_metrics = {
            'acc': 'max',
            'mse': 'min'
        }
        self.mode = modes_for_metrics[self.monitor]

    def check_es(self):
        metric_history = self.neural_net.metric_history[self.monitor][-self.patience:]
        if len(metric_history) == self.patience:
            if self.mode == "min" and all([x - metric_history[-1] < self.min_delta for x in metric_history[0:-1]]):
                return True
            elif self.mode == "max" and all([metric_history[-1] - x < self.min_delta for x in metric_history[0:-1]]):
                return True
        else:
            return False


def is_correct_class(F_nets, Y):
    if len(F_nets) > 1:
        return np.where(Y == 1)[0][0] == np.argmax([F_net for F_net in F_nets])
    elif len(F_nets) == 1:
        return Y == round(F_nets[0])


def print_progress_bar(text, a, b):
    print("{0:10}".format(text), end="\t")
    print("{0:12}".format(f"{a} of {b}"), end="\t")
    print("{0:350}".format("=" * math.ceil(30 * (a/b)) + "." * math.floor(30 * (b-a) / b)))




