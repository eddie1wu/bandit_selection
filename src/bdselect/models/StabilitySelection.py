
import numpy as np
from sklearn.linear_model import Lasso

from tqdm import tqdm

class StabilitySelection:

    def __init__(
            self,
            lam,
            num_iter = 100,
            subsample_fraction = 0.5,
            weakness = 0.5,
            pw = 0.5,
            pi_thr = 0.6,
            rng = None
    ):
        self.lam = lam

        self.num_iter = num_iter
        self.subsample_fraction = subsample_fraction

        self.weakness = weakness
        self.pw = pw
        self.pi_thr = pi_thr

        self.rng = rng


    def _normalize_columns(self, X):
        norms = np.linalg.norm(X, axis = 0)
        norms[norms == 0] = 1

        return X / norms


    def _random_lasso(self, X, y):

        n, p = X.shape

        # Draw random weights
        W = np.where(
            self.rng.uniform(size = p) < self.pw,
            self.weakness,
            1.0
        )

        # Xw = X * W
        Xw = X
        alpha_sklearn = self.lam / (2 * n)

        # Fit LASSO
        model = Lasso(alpha = alpha_sklearn, fit_intercept = False, max_iter = 20000)
        model.fit(Xw, y)

        beta_hat = model.coef_
        selected = np.flatnonzero(np.abs(beta_hat) > 1e-6)

        return selected


    def fit(self, X, y):

        X = self._normalize_columns(X)

        n, p = X.shape
        m = int(self.subsample_fraction * n)

        counts = np.zeros(p)

        selected_sets = []
        selection_prob = []

        for t in tqdm(range(self.num_iter)):

            # idx = self.rng.choice(n, size = m, replace = False)
            idx = self.rng.integers(n, size=n)

            X_sub = X[idx]
            y_sub = y[idx]

            selected = self._random_lasso(X_sub, y_sub)

            counts[selected] += 1

            selected_sets.append(selected)
            selection_prob.append(counts / (t+1))

        print("DONE running stability selection for this fixed lambda.")

        self.selected_sets_ = selected_sets
        self.selection_prob_ = selection_prob
        self.stable_set_ = np.flatnonzero(selection_prob[-1] >= self.pi_thr)

        return self


