"""
Module with class for input dataset analysis

"""
import pandas as pd
import numpy as np
import maths
# pylint: disable=C0103
# pylint: disable=W0703


class DataAnalyser:
    """
    Class designed for loading and reading a dataset in a csv format to a pd.dataframe

    """
    def __init__(self):
        self.data = None
        self.y = None
        self.x = None
        self.batches_x = []
        self.batches_y = []

    def load_data(self, path='data/data.csv', var=None, scaler=None):
        """
        Loads data from csv

        :param str scaler: path to a file with scaling metrics if exist
        :param str path: path to a csv file with dataset
        :param var:
        """
        try:
            if var:
                data = pd.DataFrame(var)
            else:
                data = pd.read_csv(path, header=None, index_col=0)
        except Exception as e:
            raise RuntimeError(e) from e
        try:
            data = self.drop_garbage(data)
            # data = self.drop_multicollinear(data, [2])
            data[1] = data[1].map({'M': 1, 'B': 0})
            data_norm = self.normalize(data, scaler)
            # data_norm = self.standardize(data, scaler)

            self.data = data_norm

            # print(self.data)
            self.y = self.get_y(data_norm)
            self.x = self.get_x(data_norm)
        except Exception as e:
            print(e)

    def normalize(self, data, scaler):
        """
        Mormalizing data by subtracting min value and dividing by (max - min) difference

        Normalizes input data using min-max algorythm data = (data - min) / (max - min)
        :param str scaler: path to a file with scaling metrics for training data (min and max)
        :param pd.DataFrame data:
        :return:
        """
        return maths.normalization(data, scaler=scaler)

    def standardize(self, data, scaler):
        """
        Normalize the input data by subtracting mean and dividing by standard deviation

        :param str scaler: path to a file with scaling metrics for training data (mean and std)
        :param pd.DataFrame data: dataset
        :return: dataframe standardized
        :rtype: pd.DataFrame
        """
        data[1:] = maths.standardize(data[1:], scaler=scaler)
        return data

    def drop_garbage(self, data):
        """
        Drops NA values

        :return: dataset
        :rtype: pd.DataFrame
        """
        # data.drop(columns=[0], inplace=True)
        data.dropna(axis=0, inplace=True)
        return data

    def drop_multicollinear(self, data, col_list=None):
        """
        Drops columns given as a list

        :return: dataframe after dropping columns
        :rtype: pd.DataFrame
        """
        if col_list is None:
            col_list = []
        return data.drop(col_list, axis=1)

    @staticmethod
    def get_y(data):
        """
        Returns y vector

        :param pd.DataFrame data:
        :return: y
        """
        return data[1]

    @staticmethod
    def get_x(data):
        """
        Returns features (x) data
        :param pd.DataFrame data:
        :return: x
        """
        data_x = data.drop(labels=1, axis=1)
        return data_x

    def transform_data(self):
        """
        Transforming x, y fields to numpy arrays

        :return: x, y as np.ndarray
        """
        x = self.get_x(self.data)
        self.x = x.to_numpy()
        y = self.get_y(self.data)
        self.y = y.to_numpy()
        # self.y = np.array([abs((y_ - 1)), y_]).astype('float32')
        return x, y

    @staticmethod
    def split_to_batches(x: np.ndarray, y: np.ndarray, num_batches: int = 1, items: str ='rows'):
        """
        Splits x and y to a given number of batches

        :param np.ndarray x: x data as np.ndarray
        :param np.ndarray y: y data as np.ndarray
        :param int num_batches: a desired numeber of batches to split the datd
        :param str items: 'rows' or 'columns' to split
        :return: tuple of lists of batches
        """
        m = x.shape[0]
        mask = np.random.permutation(m)
        data_x = x[mask]
        data_y = y[mask]
        if items == 'rows':
            batch_size = int(data_x.shape[0] / num_batches)
        else:
            batch_size = int(data_x.shape[1] / num_batches)
        batches_x = []
        batches_y = []
        if items == 'columns':
            for i in range(0, num_batches):
                if i != num_batches - 1:
                    batches_x.append(data_x[:, i * batch_size:(i + 1) * batch_size])
                    batches_y.append(data_y[:, i * batch_size:(i + 1) * batch_size])
                else:
                    batches_x.append(data_x[:, i * batch_size:])
                    batches_y.append(data_y[:, i * batch_size:])
        elif items == 'rows':
            for i in range(0, num_batches):
                if i != num_batches - 1:
                    batches_x.append(data_x[i * batch_size:(i + 1) * batch_size, :])
                    batches_y.append(data_y[i * batch_size:(i + 1) * batch_size, :])
                else:
                    batches_x.append(data_x[i * batch_size:, :])
                    batches_y.append(data_y[i * batch_size:, :])
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
        x_train = train_set.iloc[:, 1:].to_numpy()
        y_train = train_set.iloc[:, 0]
        y_train = np.array([abs((y_train - 1)), y_train]).astype('float32').T
        x_test = test_set.iloc[:, 1:].to_numpy()
        y_test = test_set.iloc[:, 0]
        y_test = np.array([abs((y_test - 1)), y_test]).astype('float32').T
        return x_train, y_train, x_test, y_test
