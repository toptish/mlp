"""
Module with different functions for normalizing input datasets (nomalization/ standartization)
"""
import copy

import numpy as np
import pandas as pd


def normalization(data: pd.DataFrame, axis=1, scaler=None) -> pd.DataFrame:
    """
    Applies min-max scaling by column

    :param data: initial dataframe
    :param axis: 0 - for rows, 1 - for columns
    :return: min-max scaled DataFrame
    """
    if axis == 0:
        data = data.T
    df_min_max_scaled = data.copy()
    if scaler:
        load_scaler = np.load(scaler, allow_pickle=True).item()
        min_val = load_scaler['min'].astype(float)
        max_val = load_scaler['max'].astype(float)
    else:
        min_val = df_min_max_scaled.min().astype(float)
        max_val = df_min_max_scaled.max().astype(float)
    df_min_max_scaled = (df_min_max_scaled - min_val) / (max_val - min_val)
    scaler = {'min': min_val, 'max': max_val}
    np.save('data/scaler', scaler)
    return df_min_max_scaled


def standardize(data: pd.DataFrame, axis=1, scaler=None) -> pd.DataFrame:
    """
    Standardization is a useful technique to transform attributes with a Gaussian distribution
    and differing means and standard deviations to a standard Gaussian distribution with a mean
    of 0 and a standard deviation of 1.

    :param data: initial dataframe
    :param axis: 0 - for rows, 1 - for columns
    :return: standardized dataframe
    """

    if axis == 0:
        data = data.T
    df_norm = copy.deepcopy(data)
    if scaler:
        load_scaler = np.load(scaler, allow_pickle=True).item()
        mean_val = load_scaler['mean'].astype(float)
        std_val = load_scaler['std'].astype(float)
    else:
        mean_val = df_norm.mean().astype(float)
        std_val = df_norm.std().astype(float)
    df_norm = (df_norm - mean_val) / std_val
    df_norm[1] = data[1]
    print(df_norm)
    return df_norm
