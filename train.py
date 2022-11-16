"""
Program designed for training and saving a neural network model given a set of data (.csv format)
"""
from data_analyser import DataAnalyser
from maths import loss
from mlp import MLP
from arg_parser import ReadArguments
from plot import plot_loss


def train(data, model, options, logs=True):
    """
    Function takes data tuple (x_train, y_train, x_test, y_test), MLP object, options as input.
    iterates through epochs. In one epoch splits x_train and y_train into batches (from options.batches).
    Iterates through batches calling model.forward(x) and model.backward(x, y, learning_rate)

    :param data: (tuple) x_train, y_train, x_test, y_test
    :param model: (MLP()) MLP class object
    :param epochs: (int) number of epochs
    :param learning_rate: (float) desired learning rate
    :param logs: (bool) True for printing training and test losses
    :param batches: (int) number of batches
    :return:
    """
    try:
        x_train, y_train, x_test, y_test = data
        training_loss = []
        testing_loss = []
        for i in range(options.epochs):
            batches_x, batches_y = DataAnalyser.split_to_batches(x_train, y_train, options.batches)
            for X, y in zip(batches_x, batches_y):
                model.forward(X)
                model.backward(X, y, options.learning_rate)
            y_ = model.forward(x_train, predict=True, act_func_name=options.activation_func)
            y_t = model.forward(x_test, predict=True, act_func_name=options.activation_func)
            l_train = loss(y_train, y_)
            l_test = loss(y_test, y_t)
            training_loss.append(l_train)
            testing_loss.append(l_test)
            if logs:
                print(f'Epoch {i}, train loss {l_train:.4f}, test loss {l_test:.4f}')
        model.save(path=options.model_filename)
        return training_loss, testing_loss
    except Exception as e:
        raise RuntimeError(f'Error encountered during training phase: {e}')


def main():
    """

    :return:
    """
    try:
        options = ReadArguments()
        options.parse_arguments()
        data = DataAnalyser()
        data.load_data()
        input_dims = data.x.shape[1]
        # print(data.x.shape[1])
        output_dims = 2
        dense_layers = options.layers
        model = MLP(input_dims, dense_layers, output_dims)
        print(model.layers)
        loss = train(data.train_test_split(), model, options)
        train_loss, test_loss = loss
        if options.loss:
            plot_loss(train_loss, test_loss)

        print(f'Model - {model.layers}\n',
              f'Epochs - {options.epochs}',
              f'Learning rate - {options.learning_rate}',
              f'Model - {model.activation_outputs[2].shape}'
              )
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()


