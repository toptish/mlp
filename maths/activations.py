"""
Module with different activation functions (sigmoid, softmax, tanh, relu, leaky_relu)
"""
import numpy as np


def softmax(value, axis=1):
    """

    :param value: numpy array 2-d
    :param axis: axis to sum along
    :return:
    """
    e_value = np.exp(value - np.max(value))
    ans = e_value / np.sum(e_value, axis=axis, keepdims=True)
    return ans


def sigmoid(value, deriv=False):
    """
    A sigmoid function is a mathematical function having a characteristic
    "S"-shaped curve or sigmoid curve.

    :param deriv: bool value. True if derivative of a function is calculated
    :param value: numpy array of data
    :return: sigmoid(value)
    """

    if deriv:
        return sigmoid(value) * (1 - sigmoid(value))
        # return np.multiply(sigmoid(value), 1 - sigmoid(value))
    sigm = 1.0 / (1.0 + np.exp(-value))
    return sigm


def relu(value, deriv=False):
    """
    The rectified linear activation function or ReLU is a non-linear function or piecewise
    linear function that will output the input directly if it is positive, otherwise,
    it will output zero.

    :param value: numpy array of data
    :param deriv: bool value. True if derivative of a function is calculated
    :return: relu(value)
    """

    if deriv:
        return 1. * (value > 0)
        # return np.where(value > 0, value, 0)
    return np.maximum(0, value)
        # return np.where(value > 0, 1, 0)


def leaky_relu(value, slope=0.1, deriv=False):
    """
    Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation
    function based on a ReLU, but it has a small slope for negative values
    instead of a flat.

    :param slope: slope for value < 0
    :param deriv: bool value. True if derivative of a function is calculated
    :param value: numpy array of data
    :return: leaky_relu(value)
    """

    if deriv:
        return np.maximum(slope, 1. * (value > 0))
    l_relu = np.maximum(slope * value, value)
    return l_relu


def tanh(value, deriv=False):
    """
    Y = tanh( X ) returns the hyperbolic tangent of the elements of X .

    :param deriv: bool value. True if derivative of a function is calculated
    :param value: numpy array of data
    :return: tanh(x)
    """

    if deriv:
        return 1. - np.tanh(value) ** 2
    return np.tanh(value)


# def stable_softmax(x, deriv=False, axis=0):
#     """
#
#     :param x:
#     :param deriv:
#     :param axis:
#     :return:
#     """
#     if deriv:
#         exp = np.exp(x)
#         exp_sum = np.sum(exp, axis=axis, keepdims=True)
#         sm = exp / exp_sum
#         derivative = np.multiply(sm, (1 - sm))
#         return derivative
#     exps = np.exp(x - np.max(x))
#     return exps / np.sum(exps, axis=axis, keepdims=True)


ACTIVATION_MAP = {
    'relu': relu,
    'leaky_relu': leaky_relu,
    'tanh': tanh,
    'sigmoid': sigmoid,
    # 'softmax': softmax,
    # 'stable_softmax': stable_softmax
}
