"""

"""
import numpy as np


def softmax(a, axis=1):
    """

    :param a:
    :param axis:
    :return:
    """
    e_a = np.exp(a)
    ans = e_a / np.sum(e_a, axis=axis, keepdims=True)
    return ans


def sigmoid(value, deriv=False):
    """
    Sigmoid function

    :param deriv:
    :param value:
    :return:
    """

    if deriv:
        return sigmoid(value) * (1 - sigmoid(value))
        # return np.multiply(sigmoid(value), 1 - sigmoid(value))
    else:
        sigm = 1.0 / (1.0 + np.exp(-value))
        return sigm


def relu(value, deriv=False):
    """

    :param deriv:
    :param value:
    :return:
    """

    if deriv:
        return 1. * (value > 0)
        # return np.where(value > 0, value, 0)

    else:
        return np.maximum(0, value)
        # return np.where(value > 0, 1, 0)


def leaky_relu(value, slope=0.1, deriv=False):
    """

    :param slope:
    :param deriv:
    :param value:
    :return:
    """

    if deriv:
        np.maximum(slope, 1. * (value > 0))
    else:
        return np.maximum(slope * value, value)


def tanh(value, deriv=False):
    """

    :param value:
    :return:
    """

    if deriv:
        return 1. - np.tanh(value) ** 2
    else:
        return np.tanh(value)


def stable_softmax(x, deriv=False, axis=0):
    if deriv:
        exp = np.exp(x)
        sum = np.sum(exp, axis=axis, keepdims=True)
        sm = exp / sum
        derivative = np.multiply(sm, (1 - sm))
        return derivative
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=axis, keepdims=True)


ACTIVATION_MAP = {
    'relu': relu,
    'leaky_relu': leaky_relu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'softmax': softmax,
    'stable_softmax': stable_softmax
}
