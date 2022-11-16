"""
Program that predicts results given the path for a model
"""
import numpy as np
from maths import precision_score, recall_score, f1_score, accuracy_score
from arg_parser import ReadArguments
from mlp import Net, MLP
from data_analyser import DataAnalyser
from maths import loss
from maths import one_hot


def predict(network, data, options):
    """

    :param network:
    :param data:
    :param options:
    :return:
    """
    layers = network.layers
    mlp = MLP(layers[0], [layers[1], layers[2]], layers[3])
    mlp.load(path=options.model_filename)
    # print(mlp.model)
    data.transform_data()
    y = data.y.astype('int')
    # print(y)
    predicted = mlp.predict(data.x)
    # print(predicted)
    forecasted = mlp.forward(data.x, predict=True)
    if options.metrics:
        print(
            f'Precision - '
            f'{round(precision_score(y, predicted), 4)};'
            f' Recall - '
            f'{round(recall_score(y, predicted), 4)};'
            f' F1 - '
            f'{round(f1_score(y, predicted), 4)}'
            f' Accuracy - '
            f'{round(accuracy_score(y, predicted), 4)};'
        )
    if options.loss:
        y_binary = one_hot(y, 2)
        l = loss(y_binary, forecasted)
        print(f'Cross entropy loss - {l}')


def main():
    """

    :return:
    """
    options = ReadArguments()
    options.parse_arguments()
    network = Net()
    data = DataAnalyser()
    data.load_data(path=options.file_name)
    network.load(path=options.model_filename)
    # print(network)
    predict(network, data, options)


if __name__ == '__main__':
    main()
