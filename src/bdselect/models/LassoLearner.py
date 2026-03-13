import numpy as np

from sklearn.linear_model import Lasso

class LassoLearner:

    def __init__(self, lam, rng):
        self.lam = lam
        self.rng = rng


    def _normalize_columns(self, X):
        norms = np.linalg.norm(X, axis = 0)
        norms[norms == 0] = 1

        return X / norms


    def fit(self, X, y, subset_idx):

        if len(subset_idx) == 0:
            self.selected_idx_ = []
            return self

        n, p = X.shape

        # Bootstrap and take subset
        bs_sample = self.rng.integers(n, size = n)
        # bs_sample = self.rng.choice(n, size = int(n/2), replace=False)
        X = X[bs_sample, :]
        X_sub = X[:, subset_idx]
        y = y[bs_sample]

        # Normalize to unit norm
        X_sub = self._normalize_columns(X_sub)

        # Fit LASSO
        alpha_sklearn = self.lam / (2.0 * n)
        model = Lasso(
            alpha = alpha_sklearn,
            fit_intercept = False,
            max_iter = 20000
        )
        model.fit(X_sub, y)

        nonzero_idx = np.flatnonzero( np.abs(model.coef_) > 1e-6 )

        self.selected_idx_ = subset_idx[nonzero_idx]

        return self

