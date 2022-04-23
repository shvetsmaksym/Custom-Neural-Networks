import numpy as np


def one_hot_encoding(y, classes):
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
    test_x, test_y = test_data[:, :len(X.T)], test_data[:, len(X.T):]

    return train_x, train_y, test_x, test_y


def check_if_classes_balanced(y, max_gap_between_classes=0.03):
    """Calculate the gap between the most popular and least popular classes in y in percents.
    Return True if classes are balanced (the calculated gap < max_gap_between_classes).
    Return False otherwise."""
    if y is None:
        return False

    p_ = [int(np.where(y_ == 1)[0]) for y_ in y]
    cl_ratio = [p_.count(u) / len(p_) for u in set(p_)]

    if max(cl_ratio) - min(cl_ratio) < max_gap_between_classes:
        return True
    else:
        print(f"Max gap between classes is {round(max(cl_ratio) - min(cl_ratio), 2)}. Need to shuffle again.")
        return False









