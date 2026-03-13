import numpy as np

from sklearn.model_selection import train_test_split

from tqdm import tqdm

class BanditSelector:

    def __init__(
            self,
            prior_alpha,
            prior_beta,
            pi_thr,
            rng
    ):

        self.alpha = prior_alpha
        self.beta = prior_beta
        self.p = len(prior_alpha)   # the total feature dimension
        self.pi_thr = pi_thr

        self.rng = rng

    def sample_subset(self):

        theta = self.rng.beta(self.alpha, self.beta)
        subset_idx = np.argwhere(theta >= 0.5).reshape(-1)

        return subset_idx

    def update(self, sucess_arms, fail_arms):
        self.alpha[sucess_arms] += 1
        self.beta[fail_arms] += 1

    def recommend(self):

        self.selection_prob_ = self.alpha / (self.alpha + self.beta)

        return np.flatnonzero(self.selection_prob_ >= self.pi_thr)



