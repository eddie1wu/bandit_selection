import numpy as np

from sklearn.datasets import make_regression

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

def make_linreg(
        n_samples,
        n_features,
        n_informative,
        intercept=0,
        std_error = 1,
        specify_coef = None,
        return_coef = False,
        random_state = None
):

    X, y, true_coef = make_regression(
        n_samples = n_samples,
        n_features = n_features,
        n_informative = n_informative,
        bias = intercept,
        noise = std_error,  # std error
        shuffle = False,  # So that the first k columns are informative
        coef = True,
        random_state = random_state
    )

    if specify_coef is not None:
        y = X @ specify_coef + intercept + np.random.normal(0, std_error, n_samples)
        true_coef = specify_coef

    return (X, y, true_coef) if return_coef else (X, y)


def make_friedman(
        n_samples,
        n_features,
        std_error,
        dgp = 1, # either 1 or 2
        random_state = None
):

    if random_state is not None:
        np.random.seed(random_state)

    X = np.random.uniform(size = (n_samples, n_features))

    if dgp == 1:
        fX = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 10 * X[:, 3] + 5 * X[:, 4]
    elif dgp == 2:
        fX = 0.1 * np.exp(4 * X[:,0]) + 4/(1 + np.exp(-20 * (X[:,1] - 0.5))) + 4*X[:,2] + 3*X[:,3] + 2*X[:,4]

    y = fX + np.random.normal(0, std_error, n_samples)

    return (X, y)


def make_lianglizhou(
        n_samples,
        n_features,
        std_error = 0.5,
        random_state = None
):

    if random_state is not None:
        np.random.seed(random_state)

    Z = np.random.randn(n_samples, n_features)
    E = np.random.randn(n_samples, 1)

    X = (E+Z)/2

    fX = 10*X[:,1] / (1 + X[:, 0]**2) + 5*np.sin(X[:,2] * X[:,3]) + 2*X[:,4]

    y = fX + np.random.normal(0, std_error, n_samples)

    return (X, y)


def plot_posterior_mean(out, time, q, title, save_name):
    time_range = np.arange(time)
    out_array = np.array(out)

    # Plot each variable
    plt.figure()
    for i in range(out_array.shape[1]):
        if i <= q-1:
            plt.plot(time_range, out_array[:, i], color = 'red', alpha = 1)
        else:
            plt.plot(time_range, out_array[:, i], color = 'blue', alpha = 0.4)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("pi_k")
    plt.yticks(np.arange(0, 1.1, 0.05))

    plt.savefig(save_name, dpi=400)
    plt.show()
    plt.close()


def plot_statistic(result_dict, title, save_name):
    # Extract methods and values for plots
    methods = list(result_dict.keys())
    values = list(result_dict.values())

    plt.bar(methods, values)
    plt.xticks(rotation=10)
    plt.xlabel('Algorithm')
    plt.ylabel('Time to Convergence (0.99)')
    plt.title(title)

    plt.savefig(save_name, dpi=400)
    plt.close()

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row.values[:-1], dtype=torch.float32)  # All columns except last are features
        label = torch.tensor(row.values[-1], dtype=torch.float32)  # Last column is label, integer
        return features, label
