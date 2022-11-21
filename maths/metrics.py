"""
Module with key machine learning models metrics
"""
import numpy as np


def compute_tp_tn_fp_fn(y_actual: np.ndarray, predicted: np.ndarray):
    """
    Takes y actual and predicted as input and returns numbers:
        TruePositive, TrueNegative, FalsePositive, FalseNegative

    :param y_actual: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (TruePositive, TrueNegative, FalsePositive, FalseNegative)
    """
    true_positive = np.sum((y_actual == 1) & (predicted == 1))
    true_negative = np.sum((y_actual == 0) & (predicted == 0))
    false_positive = np.sum((y_actual == 0) & (predicted == 1))
    false_negative = np.sum((y_actual == 1) & (predicted == 0))
    return true_positive, true_negative, false_positive, false_negative


def accuracy_score(y_actual: np.ndarray, predicted: np.ndarray):
    """
    Accuracy = (TP + TN) / (FP + FN + TP + TN)
    :param y_actual: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) accuracy in decimal
    """
    return np.mean(predicted == y_actual)


def precision_score(y_actual: np.ndarray, predicted: np.ndarray):
    """
    Precision = TP / (TP + FP)
    :param y_actual: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) precision in decimal
    """
    t_p, _, f_p, _ = compute_tp_tn_fp_fn(y_actual, predicted)
    return t_p / float(t_p + f_p)


def recall_score(y_actual, predicted):
    """
    Recall = TP / (FN + TP)
    :param y_actual: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) recall in decimal
    """
    t_p, _, _, f_n = compute_tp_tn_fp_fn(y_actual, predicted)
    return t_p / (f_n + t_p)


def f1_score(y_actual: np.ndarray, predicted: np.ndarray):
    """
    Harmonic mean of precision and recall
    :param y_actual: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) F1 score in decimal
    """
    precision = precision_score(y_actual, predicted)
    recall = recall_score(y_actual, predicted)
    f1_metric = (2 * precision * recall) / (precision + recall)
    return f1_metric
