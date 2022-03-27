import numpy as np
from DataStructures.Neurons import Perceptron

# Define training dataset
data = np.array([[-3, -4, -1],
                 [-2, 1, -1],
                 [0, 1, -1],
                 [2, 2, -1],
                 [-2, -4, 1],
                 [0, -2, 1],
                 [2, 1, 1],
                 [3, -4, 1]]
                )

# Insert bias column into training dataset
data = np.insert(data, 0, np.array([1 for _ in range(8)]), axis=1)

# Define weights array
weights = np.array([-1, -1, -1])


if __name__ == '__main__':
    perceptron = Perceptron(data=data, weights=weights, thresh=0, outputs=(1, -1), speed=2)
    perceptron.learn_perceptron()
    perceptron.show_plot()
