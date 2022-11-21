"""
Module with functions for loss calculations
"""
import numpy as np


def loss(y_oht, y_actual, axis=1) -> float:
    """
    Cross-entropy loss for multiclass classification

    :param  np.ndarray y_oht: one-hot encoded predicted value
    :param np.ndarray y_actual: actual value
    :return: cross-entropy loss
    """
    dot = y_oht * np.log(y_actual + 1e-7)
    # print(np.sum(dot, axis=axis))
    return -np.mean(np.sum(dot, axis=axis))
    # return -np.mean(y_oht * np.log(y_actual + 1e-7))
