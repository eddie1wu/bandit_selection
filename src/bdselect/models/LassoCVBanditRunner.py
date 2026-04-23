import numpy as np

from tqdm import tqdm

class LassoBanditRunner:

    def __init__(self, selector, learner):
        self.selector = selector
        self.learner = learner

    def run(self, X, y, T):

        selected_sets = []
        selection_prob = []
        chosen_lambda = []

        for t in tqdm(range(T)):

            # Choose subset
            subset = self.selector.sample_subset()

            # Fit model and compute reward
            self.learner.fit(X, y, subset)

            # Update
            success_arms = self.learner.selected_idx_
            fail_arms = np.setdiff1d(subset, success_arms)
            self.selector.update(success_arms, fail_arms)

            # Save selected set and selection probabilities
            selected_sets.append(success_arms)
            _ = self.selector.recommend()
            selection_prob.append(self.selector.selection_prob_)

            # Save lambda chosen by CV in this iteration
            chosen_lambda.append(self.learner.best_lambda_)

        self.selected_sets_ = selected_sets
        self.selection_prob_ = selection_prob
        self.chosen_lambda_ = chosen_lambda
        self.stable_set_ = self.selector.recommend()

        return self