import numpy as np
from sklearn.linear_model import LassoCV


class LassoLearnerCV:

    def __init__(
        self,
        lambda_grid,
        rng,
        cv=5,
        max_iter=20000,
        tol=1e-4,
        coef_tol=1e-6,
        use_1se=True,
    ):
        self.lambda_grid = np.asarray(lambda_grid)
        self.rng = rng
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.coef_tol = coef_tol
        self.use_1se = use_1se

    def _normalize_columns(self, X):
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1.0
        return X / norms

    def fit(self, X, y, subset_idx):

        if len(subset_idx) == 0:
            self.selected_idx_ = np.array([], dtype=int)
            self.best_lambda_ = None
            return self

        n, p = X.shape

        bs_sample = self.rng.integers(n, size=n)
        X = X[bs_sample, :]
        X_sub = X[:, subset_idx]
        y = np.asarray(y[bs_sample]).reshape(-1)

        X_sub = self._normalize_columns(X_sub)

        # Try using the grid directly as sklearn alphas first
        # instead of dividing by 2n
        alpha_grid = np.asarray(self.lambda_grid, dtype=float)

        model = LassoCV(
            alphas=alpha_grid,
            cv=self.cv,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        model.fit(X_sub, y)

        if self.use_1se:
            mse_mean = model.mse_path_.mean(axis=1)
            mse_se = model.mse_path_.std(axis=1, ddof=1) / np.sqrt(model.mse_path_.shape[1])

            i_min = np.argmin(mse_mean)
            cutoff = mse_mean[i_min] + mse_se[i_min]

            # sklearn stores alphas in descending order
            eligible = np.where(mse_mean <= cutoff)[0]
            i_star = eligible[0]   # largest alpha within 1-SE rule
            alpha_star = model.alphas_[i_star]

            coef_path = model.path(X_sub, y, alphas=model.alphas_, max_iter=self.max_iter)[1]
            coef = coef_path[:, i_star]
        else:
            alpha_star = model.alpha_
            coef = model.coef_

        nonzero_idx = np.flatnonzero(np.abs(coef) > self.coef_tol)

        self.selected_idx_ = np.asarray(subset_idx)[nonzero_idx]
        self.best_alpha_sklearn_ = alpha_star
        self.best_lambda_ = alpha_star
        self.model_ = model

        return self



# import numpy as np
# from sklearn.linear_model import LassoCV
#
#
# class LassoLearnerCV:
#
#     def __init__(
#         self,
#         lambda_grid,
#         rng,
#         cv=5,
#         max_iter=20000,
#         tol=1e-4,
#         coef_tol=1e-4,
#     ):
#         self.lambda_grid = np.asarray(lambda_grid)
#         self.rng = rng
#         self.cv = cv
#         self.max_iter = max_iter
#         self.tol = tol
#         self.coef_tol = coef_tol
#
#     def _normalize_columns(self, X):
#         norms = np.linalg.norm(X, axis=0)
#         norms[norms == 0] = 1.0
#         return X / norms
#
#     def fit(self, X, y, subset_idx):
#
#         if len(subset_idx) == 0:
#             self.selected_idx_ = np.array([], dtype=int)
#             self.best_lambda_ = None
#             return self
#
#         n, p = X.shape
#
#         # Bootstrap and take subset, same as before
#         bs_sample = self.rng.integers(n, size=n)
#         X = X[bs_sample, :]
#         X_sub = X[:, subset_idx]
#         y = np.asarray(y[bs_sample]).reshape(-1)
#
#         # Normalize to unit norm
#         X_sub = self._normalize_columns(X_sub)
#
#         # Convert your lambda grid to sklearn alpha grid
#         # sklearn objective:
#         # (1/(2n)) ||y - Xb||^2 + alpha ||b||_1
#         alpha_grid = self.lambda_grid / (2.0 * n)
#
#         model = LassoCV(
#             alphas=alpha_grid,
#             cv=self.cv,
#             fit_intercept=False,
#             max_iter=self.max_iter,
#             tol=self.tol,
#             random_state=int(self.rng.integers(0, 2**31 - 1)),
#         )
#         model.fit(X_sub, y)
#
#         nonzero_idx = np.flatnonzero(np.abs(model.coef_) > self.coef_tol)
#
#         self.selected_idx_ = np.asarray(subset_idx)[nonzero_idx]
#         self.best_alpha_sklearn_ = model.alpha_
#         self.best_lambda_ = 2.0 * n * model.alpha_
#         self.model_ = model
#
#         return self
#
#
