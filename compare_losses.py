"""
Program that reads multiple lines with training options from data/comp.txt and saves losses
to data/compare_dfs.csv

"""

import warnings
import pandas as pd
from data_analyser import DataAnalyser
from mlp import MLP
from train import train
from arg_parser import ReadArguments
from maths import ACTIVATION_MAP
# pylint: disable=W0703

warnings.filterwarnings("ignore")



def read_input():
    """
    Reads the training options from a command line

    :return: list of models
    """
    model_list = []
    dataset_path = input("Enter path to a dataset file:\n")
    try:
        dataset = DataAnalyser()
        dataset.load_data(dataset_path)
        data = dataset.train_test_split()
    except Exception as error:
        print(error)
        return model_list
    # try:
    while True:
        params = input("Enter training params separated with ';'\n"
                       "dense Layer 1 neurons; "
                       "dense layer 2 neurons; "
                       "activation function; "
                       "number of batches;"
                       "learning rate;"
                       # "bool True or False for early stopping;"
                       "number of epochs to train\n"
                       "Double click enter to stop.\n")
        if not params:
            return model_list
        param_list = params.strip().split(";")
        param_list = [item.strip() for item in param_list]
        layer1, layer2, _, batches, learn_rate, epochs = param_list
        if validate_input(param_list):
            model_dict = list_to_dict(param_list)
            options = ReadArguments()
            options.set_fields(int(layer1),
                               int(layer2),
                               float(learn_rate),
                               int(batches),
                               int(epochs))
            train_loss, val_loss, mlp_obj = train_and_get_losses(model_dict, options, data)
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['model'] = mlp_obj
            model_dict['name'] = "; ".join(param_list)
            model_list.append(model_dict)
        else:
            print("Please check the model params")
    return model_list


def validate_input(param_list: list):
    """
    Validates the training option list


    :param param_list: a list of options [layer1 neurons, layer2 neurons,
        activation function name, number of batches, learning rate, epochs]
    :return: True if all options are valid else False
    """
    try:
        layer1, layer2, act_func, batches, learn_rate, epochs = param_list
        if (int(layer1) < 3 or int(layer1) > 40) or (int(layer2) < 3 or int(layer2) > 40):
            return False
        if act_func not in ACTIVATION_MAP.keys():
            return False
        if int(batches) < 1 or int(batches) > 30:
            return False
        if float(learn_rate) >= 1 or float(learn_rate) < 0.00001:
            return False
        if int(epochs) < 1 or int(epochs) > 10000:
            return False
        if int(batches) < 1 or int(batches) >= 30:
            return False
        return True
    except Exception:
        return False


def list_to_dict(param_list: list) -> dict:
    """
    Transforms valid training options list to a dictionary

    :param list param_list: list of training options (valid)
    :return: training options dictionary
    """
    param_dict = {}
    param_dict['layers'] = [int(param_list[0]), int(param_list[1])]
    param_dict['activation'] = param_list[2]
    param_dict['batches'] = int(param_list[3])
    param_dict['learning_rate'] = float(param_list[4])
    param_dict['epochs'] = int(param_list[5])
    return param_dict


def train_and_get_losses(model_dict: dict, options: ReadArguments, data: tuple) -> tuple:
    """
    Trains the model with options and params given and returns training loss list and validation loss list

    :param dict model_dict: a dictionary with model params
    :param ReadArguments options: options (ReadArgumnets object)
    :param tuple data: (x_train, y_train, x_test, y_test)
    :return: (train loss, validation loss)
    """
    x_train, y_train, _, _ = data
    mlp = MLP(x_train.shape[1],
              model_dict['layers'],
              y_train.shape[1],
              model_dict['activation']
              )
    train_loss, val_loss = train(data, mlp, options, False)
    return train_loss, val_loss, mlp


def model_to_df(model: dict) -> pd.DataFrame:
    """
    Transforms model dict to a dataframe

    :param dict model:
    :return: model train and validation results in a dataframe
    """
    df_model = pd.DataFrame([model['train_loss'], model['val_loss']]).T
    df_model.rename(columns={0: 'train_loss', 1: 'val_loss'}, inplace=True)
    df_model['activation'] = model['activation']
    df_model['name'] = model['name']
    df_model['batches'] = model['batches']
    df_model['learning_rate'] = model['learning_rate']
    df_model.index.name = 'epoch'
    df_model = df_model.reset_index()
    return df_model


def get_dataframes(models: list) -> pd.DataFrame:
    """
    Get the dataframe from a list of dataframes

    :param list models: list of models
    :return: long dataframe of models data (train and validation
        losses, activation functions ...)
    :rtype: pd.DataFrame
    """
    df_long = pd.DataFrame(
        columns=[
            'epoch',
            'train_loss',
            'val_loss',
            'name',
            'batches'
        ])
    for model in models:
        df_model = model_to_df(model)
        df_long = df_long.append(df_model)
    return df_long


def main():
    """
    Entry point to a program. Reads parameters for multiple models
    from command line and generates a long-form dataframe with models result,
    which can be later used for plotting multiple losses.

    """
    try:
        model_list = read_input()
        models_df = get_dataframes(model_list).copy(deep=True)
        models_df.to_csv("data/compare_dfs.csv")
    except Exception as error:
        print(error)


if __name__ == '__main__':
    main()
