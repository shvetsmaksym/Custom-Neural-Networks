import pandas as pd
import numpy as np
from DataStructures.NeuralNetwork import NeuralNetwork
from DataStructures.Preprocessing import convert_y_multiclass_to_binary, normalize, split_into_train_and_test_sets

df = pd.read_csv("iris.data", names=['x1', 'x2', 'x3', 'x4', 'Class'])

Y = df.iloc[:, 4]
X = df.iloc[:, :4]
for i, unique in enumerate(Y.unique()):
    df.loc[df.Class == unique, "Class"] = i

Y = Y.to_numpy()
Y = convert_y_multiclass_to_binary(Y, classes=len(df.iloc[:, 4].unique()))
X = X.to_numpy()
X = normalize(X)

train_X, train_y, test_X, test_y = split_into_train_and_test_sets(X, Y, ratio=0.3)
del X, Y, df

model = NeuralNetwork()
model.add_input_layer(n=8, input_shape=4)
model.add_output_layer(n=3)
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=150, lr=0.1)

print("Finish.")