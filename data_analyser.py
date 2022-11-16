import argparse
import pandas as pd
import numpy as np
import math
import maths


class DataAnalyser:
    """

    """
    def __init__(self):
        """

        """
        self.data = None
        self.y = None
        self.x = None
        self.batches_x = []
        self.batches_y = []

    def load_data(self, path='data/data.csv', var=None):
        """

        :param path:
        :param var:
        :return:
        """
        try:
            if var:
                data = pd.DataFrame(var)
            else:
                data = pd.read_csv(path, header=None, index_col=0)
        except Exception as e:
            raise RuntimeError(e)
        try:
            data = self.drop_garbage(data)
            data[1] = data[1].map({'M': 1, 'B': 0})
            data_norm = self.normalize(data)
            self.data = data_norm
            self.y = self.get_y(data_norm)
            self.x = self.get_x(data_norm)
        except Exception as e:
            print(e)

    def normalize(self, data):
        """

        :param data:
        :return:
        """
        return maths.normalization(data)

    def standardize(self, data):
        """

        :param data:
        :return:
        """
        return maths.standardize(data)

    def drop_garbage(self, data):
        """

        :return:
        """
        # data.drop(columns=[0], inplace=True)
        data.dropna(axis=0, inplace=True)
        return data

    @staticmethod
    def get_y(data):
        """

        :param data:
        :return:
        """
        return data[1]

    @staticmethod
    def get_x(data):
        """

        :param data:
        :return:
        """
        data_x = data.drop(labels=1, axis=1)
        return data_x

    def transform_data(self):
        x = self.get_x(self.data)
        self.x = x.to_numpy()
        y = self.get_y(self.data)
        self.y = y.to_numpy()
        # self.y = np.array([abs((y_ - 1)), y_]).astype('float32')
        return x, y

    @staticmethod
    def split_to_batches(x: np.ndarray, y: np.ndarray, num_batches: int = 1, items: str ='rows'):
        """
        Splits
        :param x:
        :param y:
        :param num_batches:
        :param items:
        :return:
        """
        data_x = x
        data_y = y
        batch_size = int(data_x.shape[0] / num_batches) if items == 'rows' else int(data_x.shape[1] / num_batches)
        # print(batch_size)
        batches_x = []
        batches_y = []
        if items == 'columns':
            for i in range(0, num_batches):
                if i != num_batches - 1:
                    batches_x.append(data_x[:, i * batch_size:(i + 1) * batch_size])
                    batches_y.append(data_y[:, i * batch_size:(i + 1) * batch_size])
                    # batches_x.append(data_x.iloc[:, i * batch_size:(i + 1) * batch_size])
                    # batches_y.append(data_y.iloc[:, i * batch_size:(i + 1) * batch_size])
                else:
                    batches_x.append(data_x[:, i * batch_size:])
                    batches_y.append(data_y[:, i * batch_size:])
                    # batches_x.append(data_x.iloc[:, i * batch_size:])
                    # batches_y.append(data_y.iloc[])
        elif items == 'rows':
            for i in range(0, num_batches):
                if i != num_batches - 1:
                    batches_x.append(data_x[i * batch_size:(i + 1) * batch_size, :])
                    batches_y.append(data_y[i * batch_size:(i + 1) * batch_size, :])
                else:
                    batches_x.append(data_x[i * batch_size:, :])
                    batches_y.append(data_y[i * batch_size:, :])
        # self.batches_y = batches_y
        # self.batches_x = batches_x
        return batches_x, batches_y

    def train_test_split(self, frac=0.8):
        """
        Splits the dataset into a X_train, Y_train, X_test, Y_test
        :param frac: (float) fraction of train
        :return: X_train, Y_train, X_test, Y_test
        """
        shuffled = self.data.sample(frac=1)
        i = int(frac * len(self.data))
        train_set = shuffled[:i]
        test_set = shuffled[i:]
        X_train = train_set.iloc[:, 1:].to_numpy()
        Y_train = train_set.iloc[:, 0]
        Y_train = np.array([abs((Y_train - 1)), Y_train]).astype('float32').T
        X_test = test_set.iloc[:, 1:].to_numpy()
        Y_test = test_set.iloc[:, 0]
        Y_test = np.array([abs((Y_test - 1)), Y_test]).astype('float32').T
        return X_train, Y_train, X_test, Y_test