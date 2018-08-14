import numpy as np
import pandas
from sklearn import preprocessing

# Threshold coefficient.
m = 0.5


def preprocess(input_path):
    # Read data and store it in a dataframe.
    df = pandas.read_csv(input_path)

    # Convert string data columns to categorical.
    obj_columns = df.select_dtypes(['object']).columns
    df[obj_columns] = df[obj_columns].astype('category').apply(lambda x: x.cat.codes)

    # Scaling.
    data = preprocessing.scale(np.array(df))
    return data


def calc_selection_threshold(data, V):
    selection_threshold = [m * np.var(data[:, j], ddof=1) for j in range(V)]
    return selection_threshold


def calc_stats_i(data_i, V):
    mu_i = []
    mu_tilde_i = []
    sample_var_i = []

    # Loop over dimensions.
    for j in range(V):
        data_ij = data_i[:, j]
        mu_ij = np.mean(data_ij)
        mu_tilde_ij = np.median(data_ij)
        sample_var_ij = np.var(data_ij, ddof=1)

        mu_i.append(mu_ij)
        mu_tilde_i.append(mu_tilde_ij)
        sample_var_i.append(sample_var_ij)
    return mu_i, mu_tilde_i, sample_var_i


def calc_stats(data, C, V):
    # Calculate sample median, sample mean, sample variance.
    mu = []
    mu_tilde = []
    sample_var = []

    # Loop over clusters.
    for i in C:
        data_i = data[i]
        mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i, V)
        mu.append(mu_i)
        mu_tilde.append(mu_tilde_i)
        sample_var.append(sample_var_i)

    return {'mu': mu,
            'mu_tilde': mu_tilde,
            'sample_var': sample_var}


def select_dim(data_i, V, selection_threshold, mu_hat_i=None):
    """
    Select relevant dimensions for clusters, which must obey:
    s[i][j] ** 2 + (mu[i][j] - mu_tilde[i][j]) ** 2 < s_hat[i][j] ** 2
    """
    # Return a list of 0/1 values to show whether dim_j is selected.
    selected_dim = np.zeros(V)

    # Calculate relevant statistics.
    mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i, V)

    # Use medoids if given.
    if mu_hat_i is not None:
        mu_tilde_i = mu_hat_i

    # Loop over clusters to find relevant dimensions.
    for j in range(V):
        if sample_var_i[j] ** 2 + (mu_i[j] - mu_tilde_i[j]) ** 2 < selection_threshold[j] ** 2:
            selected_dim[j] = 1
    return selected_dim
