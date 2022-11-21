"""
Program designed for training and saving a neural network model given a set of data (.csv format)
"""
import copy
import numpy as np
from data_analyser import DataAnalyser
from maths import loss, metrics
from mlp import MLP
from arg_parser import ReadArguments
from plot import plot_loss
# pylint: disable=C0103
# pylint: disable=R0914
# pylint: disable=W0703


def train(data: tuple, model: MLP, options: ReadArguments, logs=True, min_delta=0.0002, es_patience=10) -> tuple:
    """
    Function takes data tuple (x_train, y_train, x_test, y_test), MLP object, options as input.
    iterates through epochs. In one epoch splits x_train and y_train into batches(options.batches).
    Iterates through batches calling model.forward(x) and model.backward(x, y, learning_rate)

    :param tuple data: x_train, y_train, x_test, y_test
    :param MLP model: MLP class object
    :param ReadArguments options: ReadArguments class with options for training (batches, eopchs,
        learning rate, early stopping)
    :param logs: True for printing training and test losses
    :type logs: bool, optional
    :return: (train losses list, validation losses list)
    :rtype: tuple
    """
    try:
        x_train, y_train, x_test, y_test = data
        training_loss = []
        testing_loss = []
        training_acc = []
        testing_acc = []
        counter = 0
        for i in range(options.epochs):
            batches_x, batches_y = DataAnalyser.split_to_batches(x_train, y_train, options.batches)
            for x, y in zip(batches_x, batches_y):
                model.forward(x)
                model.backward(x, y, options.learning_rate)
            best_model = copy.deepcopy(model.model)
            y_ = model.forward(x_train, predict=True)
            y_t = model.forward(x_test, predict=True)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            l_train = loss(y_train, y_)
            l_test = loss(y_test, y_t)
            train_acc = metrics.accuracy_score(np.argmax(y_train, axis=1),
                                               y_train_pred)
            test_acc = metrics.accuracy_score(np.argmax(y_test, axis=1),
                                              y_test_pred)
            if len(testing_loss) > 1:
                min_l_test = min(testing_loss)
                if min_l_test - l_test >= min_delta:
                    counter = 0
                    best_model = copy.deepcopy(model.model)
                else:
                    counter += 1
            if options.early_stopping and counter >= es_patience and i >= 50:

                model.model = best_model
                break
            training_loss.append(l_train)
            testing_loss.append(l_test)
            training_acc.append(train_acc)
            testing_acc.append(test_acc)
            if logs:
                print(f'Epoch {i + 1}/{options.epochs}-'
                      f' loss: {l_train:.4f} - val_loss {l_test:.4f}')
            if options.metrics:
                print(f'Epoch {i + 1}/{options.epochs}-'
                      f' train accuracy: {train_acc:.4f} - val accuracy {test_acc:.4f}')
        model.save(path=options.model_filename)
        return training_loss, testing_loss
    except Exception as e:
        raise RuntimeError(f'Error encountered during training phase: {e}') from e


def main():
    """
    Main train module function. Reads options, creates MLP options, loads datasets,
    trains model and saves the results
    """
    try:
        options = ReadArguments()
        options.parse_arguments()
        data = DataAnalyser()
        data.load_data(options.file_name)
        input_dims = data.x.shape[1]
        output_dims = 2
        dense_layers = options.layers
        model = MLP(input_dims, dense_layers, output_dims, options.activation_func)
        losses = train(data.train_test_split(), model, options)
        train_loss, test_loss = losses
        if options.loss:
            plot_loss(train_loss, test_loss)

        print(f'Model - {model.layers}\n',
              f'Epochs - {options.epochs}',
              f'Learning rate - {options.learning_rate}'
              )
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
