import numpy as np
from DataStructures.NeuralNetwork import NeuralNetwork
from DataStructures.Layers import Input, Output, Hidden

if __name__ == "__main__":
    data = np.array([[0, 0, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0]])

    train_x, train_y = data[:, :2], data[:, 2]

    model = NeuralNetwork()
    model.add_input_layer(n=2, input_shape=2)
    model.add_output_layer(n=1)
    model.fit(train_x, train_y, validation_data=(train_x, train_y), lr=0.33, epochs=10000)
    print("Finish")

