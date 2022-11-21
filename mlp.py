"""
Net class and MLP class

"""
import numpy as np
from maths import softmax
from maths import ACTIVATION_MAP
from maths import normalized_xavier_initializer, small_random_numbers
# pylint: disable=C0103
# pylint: disable=R0914


class Net:
    """
    Base class with methods for saving and loading a model

    """
    def __init__(self):
        self.model = {}
        self.layers = []

    def save(self, path='data/model.npy'):
        """
        Method for saving model weights given path as an argument.

        :param str path:
        """
        np.save(path, self.model)

    def load(self, path='data/model.npy'):
        """
        Method for loading a model.


        :param str path:
        :return: model topology (layers and activation function)
        :rtype: dict
        """
        model = np.load(path, allow_pickle=True).item()
        self.model = model
        self.layers = model['layers']
        return model


class MLP(Net):
    """
    Multilayer Perceptron class for 4 layer (1 input, 2 dense, 1 output)

    """
    def __init__(self,
                 input_size: int = 2,
                 layers=None,
                 output: int = 2,
                 act_func: str = 'relu'):
        super().__init__()
        if layers is None:
            layers = [16, 8]
        self.activation_outputs = None
        self.z_outputs = None
        self.layers = [input_size] + layers + [output]
        np.random.seed(0)
        model = {}

        model['w1'] = normalized_xavier_initializer((input_size, layers[0]))
        model['b1'] = small_random_numbers((1, layers[0]))

        model['w2'] = normalized_xavier_initializer((layers[0], layers[1]))
        model['b2'] = small_random_numbers((1, layers[1]))

        model['w3'] = normalized_xavier_initializer((layers[1], output))
        model['b3'] = small_random_numbers((1, output))

        model['layers'] = self.layers
        model['activation'] = act_func
        self.model = model

    def forward(self, X, predict=False):
        """
        Forward propagation algorithm

        :param np.ndarray X: features data
        :param bool predict: False if method is used for prediction
            (not to change activation outputs)
        :return: y array probabilities
        """
        act_func_name = self.model['activation']

        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

        z1 = np.dot(X, w1) + b1
        a1 = self.activation(act_func_name)(z1)

        z2 = np.dot(a1, w2) + b2
        a2 = self.activation(act_func_name)(z2)

        z3 = np.dot(a2, w3) + b3
        y_ = softmax(z3)

        if not predict:
            self.activation_outputs = (a1, a2, y_)
            self.z_outputs = (z1, z2, z3)
        return y_

    def backward(self, X, y, learning_rate=0.01):
        """
        Backard propagation algorythm. Applying chain derivative rule to get all
        gradients to change the initial weights and biases.

        :param X: features data
        :param y: actual y values (one-hot encoded)
        :param learning_rate: learning rate to adjust gradients (optional, default 0.01)
        """
        act_func_name = self.model['activation']

        _, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        # b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']
        a1, a2, y_ = self.activation_outputs
        z1, z2, _ = self.z_outputs
        m = X.shape[0]

        delta3 = y_ - y
        dw3 = np.dot(a2.T, delta3)
        db3 = np.sum(delta3, axis=0) / float(m)

        delta2 = self.activation(act_func_name)(z2, deriv=True) * np.dot(delta3, w3.T)
        dw2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0) / float(m)

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
        Predicts Y given a set of X using fitted model

        :param np.ndarray X: fetaures data
        :return: y-values (1 - for positive, 0 - for negative class)
        :rtype: np.ndarray
        """
        y_out = self.forward(X, predict=True)
        return np.argmax(y_out, axis=1)

    def predict_proba(self, X):
        """
        Predicts probabilities of classes

        :param np.ndarray X: features data
        :return: probabilities of y belonging to a certain class
        :rtype: np.ndarray
        """
        return self.forward(X, predict=True)


    def activation(self, function_name: str = 'tanh'):
        """
        Takes function name (str) and returns a pointer to the corresponding function.

        :param function_name: [relu, tanh, sigmoid, leaky_relu]
        :return: pointer to function
        """
        try:
            return ACTIVATION_MAP[function_name]
        except Exception as e:
            raise RuntimeError('Unknown activation function') from e
