"""
One-hot encoding function
"""
import numpy as np


def one_hot(y_value, depth):
    """
    Ebcode categorical features as a one-hot numeric array

    :param y_value: initial 1-d array
    :param depth: number of output classes / features
    :return: one-hote encoded vector
    """
    y_value = y_value.astype(int)
    num = y_value.shape[0]
    y_oht = np.zeros((num, depth))
    y_oht[np.arange(num), y_value] = 1
    return y_oht
