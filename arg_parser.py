"""
Module with classes for parsing program options and arguments
"""
import argparse


# pylint: disable=R0902
# pylint: disable=R0903
# pylint: disable=R0913

class ReadArguments:
    """
    Reads and parses command line program arguments for train program
    """

    def __init__(self):
        self.file_name = "data/data.csv"
        self.model_filename = 'model'
        self.activation_func = ""
        self.learning_rate = 0.05
        self.epochs = 300
        self.metrics = False
        self.batches = 6
        self.early_stopping = False
        self.loss = False
        self.layers = []

    def parse_arguments(self):
        """
        Has the following fields

        - file_name,
        - learning_rate,
        - number of epochs,
        - loss
        - early_stopping
        - activation_func
        - layers
        - model_filename
        - batches
        """
        parser = argparse.ArgumentParser('Multilayer perceptron input arguments')
        parser.add_argument('--learning_rate',
                            '-r',
                            action="store",
                            dest="r",
                            type=float,
                            default=0.03,
                            help='Set custom learning rate (default: 0.01)')
        parser.add_argument('--loss',
                            '-l',
                            action="store_true",
                            dest='l',
                            help='Plots loss (Entropy) change for train and test datasets'
                                 ' depending on epoch number')
        parser.add_argument('--metrics', '-m',
                            action="store_true",
                            dest="m",
                            help='Model quality metrics')
        parser.add_argument('--early_stopping',
                            '-s',
                            action="store_true",
                            dest='s',
                            help='Plots loss (Entropy) change for train and test datasets depending'
                                 ' on epoch number')
        parser.add_argument('--epochs',
                            '-e',
                            action="store",
                            type=int,
                            dest='e',
                            default=500,
                            help='Custom epochs number (default: 1000)')
        parser.add_argument('--batches',
                            '-b',
                            action="store",
                            type=int,
                            dest='b',
                            default=5,
                            help='Number of batches. Default 10')
        parser.add_argument('--dense1',
                            '-d1',
                            action="store",
                            type=int,
                            dest='d1',
                            default=16,
                            help='Neurons number on 1st dense layer)')
        parser.add_argument('--dense2',
                            '-d2',
                            action="store",
                            type=int,
                            dest='d2',
                            default=8,
                            help='Neurons number on 2nd dense layer)')
        parser.add_argument('--model_filename',
                            '-f',
                            dest='f',
                            action="store",
                            type=str,
                            default='data/model.npy',
                            help='Filename for model saving')
        parser.add_argument('--activation_function',
                            '-a',
                            dest='a',
                            action="store",
                            type=str,
                            default='relu',
                            help='Activation function name: (tanh, relu, leaky_relu, sigmoid)')
        parser.add_argument('filename',
                            type=str,
                            help='Filename for csv train dataset')

        arguments = parser.parse_args()
        self.file_name = arguments.filename
        self.learning_rate = arguments.r
        self.epochs = arguments.e
        self.early_stopping = arguments.s
        self.layers = [arguments.d1, arguments.d2]
        self.loss = arguments.l
        self.metrics = arguments.m
        self.model_filename = arguments.f
        self.batches = arguments.b
        self.activation_func = arguments.a
        self.validate_args()

    def validate_args(self):
        """
        Checks whether options learning rate and epochs belong to correct range
        """
        if self.learning_rate < 0.0001 or self.learning_rate > 1:
            raise RuntimeError('Learning rate must be within (0.0001, 1) range')
        if self.epochs > 10000 or self.epochs < 1:
            raise RuntimeError('Wrong epochs number. Range (1, 10 000')
        if self.layers[0] > 30 or self.layers[0] < 2:
            raise RuntimeError('Neurons must be within (2, 30) range')
        if self.layers[1] > 30 or self.layers[1] < 2:
            raise RuntimeError('Neurons must be within (2, 30) range')
        if self.batches > 25 or self.batches < 1:
            raise RuntimeError('Wrong batches number. Range (1, 30)')

    def set_fields(self, layer1, layer2, learning_rate, batches, epochs):
        """
        Sets the corresponding class fields

        :param layer1: number of layer 1 neurons
        :param layer2: number of layer 2 neurons
        :param learning_rate: learning rate
        :param batches: number of batches to split the dataset
        :param epochs: number of epochs to train
        """
        self.layers = [layer1, layer2]
        self.learning_rate = learning_rate
        self.batches = batches
        self.epochs = epochs


class ReadPredictArgs:
    """
    Class for reading and saving predict program arguments and options
    """

    def __init__(self):
        self.metrics = False
        self.model_filename = 'model'
        self.file_name = "data/test.csv"
        self.loss = False

    def parse_arguments(self):
        """
        Parses arguments and options from command line and sets corresponding fields values
        """
        parser = argparse.ArgumentParser('Multilayer perceptron predict input arguments')
        parser.add_argument('--model_filename',
                            '-f',
                            dest='f',
                            action="store",
                            type=str,
                            default='data/model.npy',
                            help='Filename for model saving')
        parser.add_argument('--metrics', '-m',
                            action="store_true",
                            dest="m",
                            help='Model quality metrics')
        parser.add_argument('--loss',
                            '-l',
                            action="store_true",
                            dest='l',
                            help='Calculates loss (Entropy) based on predicted and actual values')
        parser.add_argument('filename',
                            type=str,
                            help='Filename for csv train dataset')
        arguments = parser.parse_args()
        self.file_name = arguments.filename
        self.metrics = arguments.m
        self.model_filename = arguments.f
        self.loss = arguments.l
