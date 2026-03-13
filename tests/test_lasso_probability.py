import numpy as np
from scipy.stats import multivariate_normal


X = np.array([
    [2, 0, 2, 0],
    [2, 2, 0, 0],
    [2, 0, 2, 2]
])



print(np.cov(X, bias = True))




def mvn_abs_box_prob(Sigma, rhs, allow_singular=False):
    rhs = np.asarray(rhs, dtype=float)

    # For |X_i| <= rhs_i, any negative rhs_i makes the event impossible
    if np.any(rhs < 0):
        return 0.0

    lower = -rhs
    upper = rhs

    return multivariate_normal.cdf(
        upper,
        mean=np.zeros(len(rhs)),
        cov=Sigma,
        lower_limit=lower,
        allow_singular=allow_singular
    )


A = np.array([
    [2, 0, 2, 0],
    [2, 2, 0, 0],
    [-8, -1, -7, -8]
])

Sigma_before = A @ A.T



B = np.array([
    [2,0,2,0],
    [-7,0, -7, -8]
])

Sigma_after = B @ B.T


lam = 0.01
n = 4
y_before = np.array([ (n-lam/2), (n-lam/2), lam/2 ])
y_after = np.array([(n-lam/2), lam/4])


print(mvn_abs_box_prob(Sigma_before, y_before))
print(mvn_abs_box_prob(Sigma_after, y_after))
