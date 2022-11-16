"""

"""


def normalization(data, axis=1):
    """
    Applies min-max scaling by column

    :param df:
    :param axis:
    :return:
    """
    if axis == 0:
        data = data.T
    df_min_max_scaled = data.copy()
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
                df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    return df_min_max_scaled


def standardize(data, axis=1):
    """
    Standardization is a useful technique to transform attributes with a Gaussian distribution
    and differing means and standard deviations to a standard Gaussian distribution with a mean
    of 0 and a standard deviation of 1.

    :param data:
    :return:
    """

    # scaler = StandardScaler().fit(data)
    # rescaled_data = scaler.transform(data)
    # print(f'data shape is {data.shape}')

    if axis == 0:
        data = data.T
    norm = (data - data.mean()) / data.std()
    return norm
