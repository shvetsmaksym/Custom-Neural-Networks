import pandas as pd
import numpy as np
from DataStructures.NeuralNetwork import NeuralNetwork
from DataStructures.Preprocessing import convert_y_multiclass_to_binary, normalize, split_into_train_and_test_sets, \
    check_if_classes_balanced

if __name__ == "__main__":
    df = pd.read_csv("iris.data", names=['x1', 'x2', 'x3', 'x4', 'Class'])

    # Preprocessing
    Y = df.iloc[:, 4]
    X = df.iloc[:, :4]
    for i, unique in enumerate(Y.unique()):
        df.loc[df.Class == unique, "Class"] = i

    Y = Y.to_numpy()
    Y = convert_y_multiclass_to_binary(Y, classes=len(df.iloc[:, 4].unique()))
    X = X.to_numpy()
    X = normalize(X)

    train_X, train_y, test_X, test_y = None, None, None, None
    while not check_if_classes_balanced(train_y):
        # Split data until classes are pretty balanced
        train_X, train_y, test_X, test_y = split_into_train_and_test_sets(X, Y, ratio=0.4)
    del X, Y, df

    # Build and train model
    model = NeuralNetwork()
    model.add_input_layer(n=32, input_shape=4)
    model.add_output_layer(n=3)
    model.fit(train_X, train_y, validation_data=(test_X, test_y), lr=0.01, epochs=10000)

    print("Finish.")