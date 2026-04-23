# Limitations and Opportunities of Bandit Algorithms for Feature Selection

This repository hosts the replication code for "Limitations and Opportunities of Bandit Algorithms for Feature Selection".

Abstract: Feature selection is an important problem in statistical learning. This paper studies the performance of bandit-based feature selection and compares it with stability selection. Theories and simulation results show that bandit selection does not generally outperform stability selection for signal recovery and may suffer from high false positive rates, par- ticularly in settings with correlated features or omitted variables. Nevertheless, bandit methods enjoy advantages in high-dimensional settings by allocating computational effort to promising regions of feature space and by adaptively handling combinatorial subset selection when exhaustive search is infeasible. We then discuss the strong identifiability condition under which bandit-based methods consistently select the set of true features, and propose a top-two Thompson sampling variant designed for pure exploration settings. Finally, we apply our method to the empirical asset pricing study of Gu et al. (2020) and obtain similar conclusions regarding signal importance.

Running order (to be updated):

-   Numerical simulations: run_rf.py -\> run_convergence.py -\> run_online.py

-   Empirical application: data_fetch.py -\> data_merge.py -\> data_clean.py -\> data_preprocess.py -\> run_gkx.py

## File structure

The `figures` folder contains the figures in the paper.

[`BinaryBandit.py`](BinaryBandit.py) the Bernoulli bandit class.

[`data_clean.py`](data_clean.py) data cleaning step, after fetching and merging.

[`data_fetch.py`](data_fetch.py) fetch returns from CRSP.

[`data_merge.py`](data_merge.py) merge firm level covariates of Gu, Kelly and Xiu (2020) with excess returns, after fetching.

[`data_preprocess.py`](data_preprocess.py) preprocessing steps for the data, after fetching, merging and cleaning.

[`ImportanceBandit.py`](ImportanceBandit.py) contains the class for bandit feature selection.

[`MLP.py`](MLP.py) contains the most basic MLP class, in Pytorch.

[`plot_fig.py`](plot_fig.py) plots all the figures in the paper. But need to generate results using the other scripts first.

[`run_best_arm.py`](run_best_arm.py) compares the convergence rates and variances of difference methods of best-arm identification under a fixed confidence setting for stochastic MAB.

[`run_convergence.py`](run_convergence.py) compares the convergence rates of Thompson sampling to top-two Thompson sampling.

[`run_gkx.py`](run_gkx.py) run bandit feature selection on the dataset of Gu, Kelly and Xiu (2020).

[`run_online.py`](run_online.py) runs different datasets in an online setting.

[`run_predictive_check.py`](run_predictive_check.py) runs sanity checks comparing the predictive performances of models using high-dimensional covariates.

[`run_rf.py`](run_rf.py) compares bandit feature importance to random forest impurity based importance and permutation importance.

[`Sampler.py`](Sampler.py) the Sampler class which contains all the best arm identification algorithms.

[`utils.py`](utils.py) contains the utility functions for generating data and plotting.
