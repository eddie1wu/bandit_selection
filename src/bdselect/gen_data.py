import numpy as np


def gen_orthog_data(n, dim_signal, dim_X, sigma_epsilon, rng):

    X = rng.normal(loc = 0.0, scale = 1.0, size = (n, dim_X))
    beta = np.zeros((dim_X, 1))
    beta[:dim_signal] = 1.0
    y = X @ beta + rng.normal(loc = 0.0, scale = sigma_epsilon, size = (n, 1))

    return X, y, beta



def gen_factor_data(n, dim_factor, dim_signal, dim_X, sigma_eta, sigma_epsilon, rng):

    # Gen factor
    F = rng.normal(loc = 0.0, scale = 1.0, size = (n, dim_factor))

    # Gen loading
    Phi = rng.normal(loc = 0.0, scale = 1.0, size = (dim_factor, dim_X))

    # Gen X
    eta = rng.normal(loc = 0.0, scale = sigma_eta, size = (n, dim_X))
    X = F @ Phi + eta

    # Gen Y
    beta = np.zeros((dim_X, 1))
    beta[:dim_signal] = 1.0
    epsilon = rng.normal(loc = 0.0, scale = sigma_epsilon, size = (n, 1))
    Y = X @ beta + epsilon

    return X, Y, beta
