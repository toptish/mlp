"""
Program that predicts results given the path for a model
"""
from maths import precision_score, recall_score, f1_score, accuracy_score, loss, one_hot
from arg_parser import ReadPredictArgs
from mlp import Net, MLP
from data_analyser import DataAnalyser
# pylint: disable=C0103
# pylint: disable=W0703


def predict(network, data, options):
    """
    Predicts y data given a Net class object,

    :param Net network:
    :param DataAnalyser data:
    :param ReadPredictArgs options:
    """
    layers = network.layers
    mlp = MLP(layers[0], [layers[1], layers[2]], layers[3])
    mlp.load(path=options.model_filename)
    data.transform_data()
    y = data.y.astype('int')
    predicted = mlp.predict(data.x)
    forecasted = mlp.forward(data.x, predict=True)
    if options.metrics:
        print(
            f'Precision - '
            f'{round(precision_score(y, predicted), 4)};'
            f' Recall - '
            f'{round(recall_score(y, predicted), 4)};'
            f' F1 - '
            f'{round(f1_score(y, predicted), 4)};'
            f' Accuracy - '
            f'{round(accuracy_score(y, predicted), 4)};'
        )
    if options.loss:
        y_binary = one_hot(y, 2)
        cel = loss(y_binary, forecasted)
        print(f'Cross entropy loss - {cel:.4f}')
    print(f'Model topology:\n'
          f'- layers: {mlp.layers}\n'
          f'- activation: {mlp.model["activation"]}')


def main():
    """

    :return:
    """
    try:
    # options = ReadArguments
        options = ReadPredictArgs()
        options.parse_arguments()
        network = Net()
        data = DataAnalyser()
        data.load_data(path=options.file_name, scaler='data/scaler.npy')
        network.load(path=options.model_filename)
        # print(network)
        predict(network, data, options)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
