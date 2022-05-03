import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from DataStructures.NeuralNetworks import NeuralNetwork

if __name__ == "__main__":
    data = np.array([[-1, -1, 0],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [1, 1, 0]])

    train_x, train_y = data[:, :2], data[:, 2]

    # Define model
    model = NeuralNetwork(classification_problem=True)
    model.add_input_layer(n=2, input_shape=2)
    model.add_output_layer(n=1)

    # Compile model
    model.add_metrics(metrics=['mse', 'acc'])
    model.add_early_stopping(patience=1500, monitor='mse', min_delta=0.0001)

    # Fit model
    model.fit(train_x, train_y, validation_data=(train_x, train_y), lr=0.1, epochs=10000)

    # Prepare plot
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Y = np.zeros((100, 100))

    # Feed network with x1 and x2
    for i, x1_ in enumerate(x1):
        for j, x2_ in enumerate(x2):
            model.feed_forward(xi=np.array([x1_, x2_]))
            Y[i, j] = model.output_layer.neurons[0].F_net

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,)

    plt.show()

    print("Finish")

