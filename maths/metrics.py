"""
Module with key machine learning models metrics
"""
import numpy as np


def compute_tp_tn_fp_fn(y: np.ndarray, predicted: np.ndarray):
    """
    Takes y actual and predicted as input and returns numbers:
        TruePositive, TrueNegative, FalsePositive, FalseNegative
    :param y: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (TruePositive, TrueNegative, FalsePositive, FalseNegative)
    """
    tp = np.sum((y == 1) & (predicted == 1))
    tn = np.sum((y == 0) & (predicted == 0))
    fp = np.sum((y == 0) & (predicted == 1))
    fn = np.sum((y == 1) & (predicted == 0))
    return tp, tn, fp, fn


def accuracy_score(y: np.ndarray, predicted: np.ndarray):
    """
    Accuracy = (TP + TN) / (FP + FN + TP + TN)
    :param y: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) accuracy in decimal
    """
    # tp, tn, fp, fn = compute_tp_tn_fp_fn(y, predicted)
    # return ((tp + tn) / float(tp + tn + fp + fn))
    return np.mean(predicted == y)


def precision_score(y: np.ndarray, predicted: np.ndarray):
    """
    Precision = TP / (TP + FP)
    :param y: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) precision in decimal
    """
    tp, tn, fp, fn = compute_tp_tn_fp_fn(y, predicted)
    return tp / float(tp + fp)


def recall_score(y, predicted):
    """
    Recall = TP / (FN + TP)
    :param y: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) recall in decimal
    """
    tp, tn, fp, fn = compute_tp_tn_fp_fn(y, predicted)
    return tp / (fn + tp)


def f1_score(y: np.ndarray, predicted: np.ndarray):
    """
    Harmonic mean of precision and recall
    :param y: (np.ndarray) y actual data
    :param predicted: (np.ndarray) y predicted data
    :return: (float) F1 score in decimal
    """
    precision = precision_score(y, predicted)
    recall = recall_score(y, predicted)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
