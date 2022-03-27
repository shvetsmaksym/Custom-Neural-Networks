import numpy as np


def convert_y_multiclass_to_binary(y, classes):
    y_out = np.zeros((len(y), classes))
    for i, yi, in enumerate(y):
        y_out[i, yi] = 1

    return y_out


def normalize(x):
    """Normalize x by columns"""
    x = x.T
    x_norm = np.zeros(x.shape)
    for i, x_col in enumerate(x):
        norm = np.linalg.norm(x_col)
        x_norm[i] = x_col / norm

    return x_norm.T


def split_into_train_and_test_sets(X, y, ratio=0.3):
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    train_data = data[:int(1 - ratio * len(data))]
    test_data = data[int(1 - ratio * len(data)):]

    train_x, train_y = train_data[:, :len(X.T)], train_data[:, len(X.T):]
    test_x, test_y = train_data[:, :len(X.T)], train_data[:, len(X.T):]

    return train_x, train_y, test_x, test_y
