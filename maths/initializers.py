"""
Module with different initialization functions
"""
import numpy as np


def normalized_xavier_initializer(size: tuple = (0, 0)) -> np.ndarray:
    """
    Xavier's initialization is a trick to train our model to converge faster read more.
    Instead of initializing our weights with small numbers which are distributed randomly
    we initialize our weights with mean zero and variance of 2/(number of inputs + number
    of outputs)

    :param size: desired size of output array
    :return: np.ndarray of given size
    :rtype: np.ndarray

    """
    low = -np.sqrt(6. / (size[0] + size[1]))
    high = np.sqrt(6. / (size[0] + size[1]))

    return np.random.uniform(low=low, high=high, size=(size[0], size[1]))
# def xavier_initializer(size: tuple = (0, 0)):


def small_random_numbers(size: tuple = (0, 0)) -> np.ndarray:
    """
    Initializes a numpy array of a given size with numbers (-0.01, 0.01)

    :param size: desired size of output array
    :return: np.ndarray of given size
    :rtype: np.ndarray
    """
    return np.random.uniform(low=-0.1, high=0.1, size=(size[0], size[1]))


def all_zero_init(size: tuple = (0, 0)) -> np.ndarray:
    """
    Initializes a numpy array of a given size with all zeros

    :param size: desired size of output array
    :return: np.ndarray of given size
    :rtype: np.ndarray
    """
    return np.zeros(size)
