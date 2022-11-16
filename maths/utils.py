"""

"""
import numpy as np


def one_hot(y, depth):
    """

    :param y:
    :param depth:
    :return:
    """
    m = y.shape[0]
    y_oht = np.zeros((m, depth))
    y_oht[np.arange(m), y] = 1
    return y_oht
