"""
Plotting training and validation losses
"""
import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss):
    """

    :param train_loss:
    :param test_loss:
    :return:
    """
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='validation loss')
    plt.legend()

    plt.show()
