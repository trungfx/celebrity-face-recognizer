import numpy as np


def load_data(train_path=None, test_path=None):
    x_train, y_train, x_test, y_test = [], [], [], []

    if train_path:
        # load file train
        npzfile = np.load(train_path)
        x_train = npzfile['embedding']
        y_train = npzfile['label']

    if test_path:
        # load file test
        npzfile = np.load(test_path)
        x_test = npzfile['embedding']
        y_test = npzfile['label']

    # show info
    print("x_train:", len(x_train), "y_train:", len(y_train))
    print("x_test:", len(x_test), "y_test:", len(y_test))
    if len(x_train) > 0:
        print("x_train_shape:", x_train[0].shape)
    if len(x_test) > 0:
        print("x_test_shape:", x_test[0].shape)

    return x_train, y_train, x_test, y_test
