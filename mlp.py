"""

"""
import numpy as np
from maths import softmax
from maths import ACTIVATION_MAP


class Net:
    """

    """
    def __init__(self):
        """

        :param input_size:
        :param layers:
        :param output:
        """
        self.model = {}
        self.layers = []

    def save(self, path='model.npy'):
        """

        :param path:
        :return:
        """
        np.save(path, self.model)

    def load(self, path='model.npy'):
        """

        :param path:
        :return:
        """
        model = np.load(path, allow_pickle=True).item()
        self.model = model
        self.layers = model['layers']
        return model


class MLP(Net):
    """

    """
    def __init__(self, input_size:int = 2, layers: list = [16, 8], output: int = 2, act_func: str = 'tanh'):
        """

        :param input_size:
        :param layers:
        :param output:
        :param act_func:
        """
        super(MLP, self).__init__()
        self.layers = [input_size] + layers + [output]
        np.random.seed(0)
        model = {}

        model['w1'] = np.random.randn(input_size, layers[0])
        model['b1'] = np.zeros((1, layers[0]))

        model['w2'] = np.random.randn(layers[0], layers[1])
        model['b2'] = np.zeros((1, layers[1]))

        model['w3'] = np.random.randn(layers[1], output)
        model['b3'] = np.zeros((1, output))

        model['layers'] = self.layers
        model['activation'] = act_func
        self.model = model

    def forward(self, X, predict=False):
        """

        :param X:
        :param predict:
        :return:
        """
        act_func_name = self.model['activation']

        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

        z1 = np.dot(X, w1) + b1
        # a1 = np.tanh(z1)
        a1 = self.activation(act_func_name)(z1)

        z2 = np.dot(a1, w2) + b2
        # a2 = np.tanh(z2)
        a2 = self.activation(act_func_name)(z2)

        z3 = np.dot(a2, w3) + b3
        y_ = softmax(z3)
        #         print(f'a2.shape - {a2.shape}\n w2.shape - {w3.shape}\n b3.shape - {b3.shape}\n y_shape - {y_.shape}')

        if not predict:
            self.activation_outputs = (a1, a2, y_)
            self.z_outputs = (z1, z2, z3)
        return y_

    def backward(self, X, y, learning_rate=0.01):
        """

        :param X:
        :param y:
        :param learning_rate:
        :return:
        """
        act_func_name = self.model['activation']

        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        # b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']
        a1, a2, y_ = self.activation_outputs
        z1, z2, z3 = self.z_outputs
        m = X.shape[0]

        delta3 = y_ - y
        dw3 = np.dot(a2.T, delta3)
        db3 = np.sum(delta3, axis=0) / float(m)

        # delta2 = (1 - np.square(a2)) * np.dot(delta3, w3.T)
        delta2 = self.activation(act_func_name)(z2, deriv=True) * np.dot(delta3, w3.T)
        dw2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0) / float(m)

        # delta1 = (1 - np.square(a1)) * np.dot(delta2, w2.T)
        delta1 = self.activation(act_func_name)(z1, deriv=True) * np.dot(delta2, w2.T)
        dw1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0) / float(m)

        self.model['w1'] -= learning_rate * dw1 / m
        self.model['b1'] -= learning_rate * db1

        self.model['w2'] -= learning_rate * dw2 / m
        self.model['b2'] -= learning_rate * db2

        self.model['w3'] -= learning_rate * dw3 / m
        self.model['b3'] -= learning_rate * db3

    def predict(self, X):
        """

        :param X:
        :return:
        """
        y_out = self.forward(X, predict=True)
        return np.argmax(y_out, axis=1)

    def activation(self, function_name: str = 'tanh'):
        """

        :param function_name:
        :return:
        """
        try:
            return ACTIVATION_MAP[function_name]
        except Exception as e:
            raise RuntimeError('Unknown activation function')
