import numpy as np


class Metrics:
    def __init__(self, neural_net):
        self.neural_net = neural_net

        self.metrics_functions = {
            'acc': self.accuracy,
            'mse': self.MSE
        }

        self.metrics_to_use = {}

    def set_metrics_to_use(self, metrics_list: list):
        self.metrics_to_use = {metric: self.metrics_functions[metric] for metric in metrics_list}
        self.neural_net.metric_history = {m: [] for m in metrics_list}

    def calculate_metrics(self, verbose=0):
        for func in self.metrics_to_use.values():
            func(verbose)

    def accuracy(self, verbose=0):
        res = np.average(self.neural_net.basic_metrics_history['correct_classifications'][-1])
        self.neural_net.metric_history['acc'].append(res)
        if verbose:
            print(f"Accuracy: {round(res, 4)}", end="\t")

    def MSE(self, verbose=0):
        res = 1/2 * (np.average(self.neural_net.basic_metrics_history['errors'][-1]) ** 2)
        self.neural_net.metric_history['mse'].append(res)
        if verbose:
            print(f"MSE: {round(res,4)}", end="\t")


class EarlyStopping:
    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.patience = None
        self.monitor = None
        self.min_delta = None

        self.count = 0

    def set_early_stopping(self, patience, monitor, min_delta):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta

    def check_es(self, mode="min"):
        """In "min" mode, training will stop when the quantity monitored has stopped decreasing.
        In "max" mode it will stop when the quantity monitored has stopped increasing.
        """
        metric_hist = self.neural_net.metric_history[self.monitor][-self.patience:]
        if len(metric_hist) == self.patience:
            if mode == "min" and metric_hist[0] - metric_hist[-1] < self.min_delta:
                return True
            elif mode == "max" and metric_hist[-1] - metric_hist[0] < self.min_delta:
                return True
        else:
            return False




