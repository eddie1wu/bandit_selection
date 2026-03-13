import numpy as np
from pathlib import Path

from bdselect.models.BanditSelector import BanditSelector
from bdselect.models.LassoLearner import LassoLearner
from bdselect.models.LassoBanditRunner import LassoBanditRunner
from bdselect.rng import make_rng
from bdselect.io.save_load import load_config, save_config, make_run_dir, save_results
from bdselect.gen_data import gen_factor_data, gen_orthog_data


def main():

    # Load config
    config_path = Path("configs/lasso_bandit.toml")
    cfg = load_config(config_path)
    rng = make_rng(cfg.seed)

    # # Gen data
    # X, y, _ = gen_factor_data(
    #     n=cfg.n_train,
    #     dim_factor=cfg.dim_factor,
    #     dim_signal=cfg.dim_signal,
    #     dim_X=cfg.dim_X,
    #     sigma_eta=cfg.sigma_eta,
    #     sigma_epsilon=cfg.sigma_epsilon,
    #     rng=rng
    # )

    X, y, _ = gen_orthog_data(
        cfg.n_train,
        cfg.dim_signal,
        cfg.dim_X,
        cfg.sigma_epsilon,
        rng
    )

    n, p = X.shape

    # Loop over the lambda grid
    lambda_list = np.logspace(-1, 2, 50)
    # lambda_list = [5]
    result_list = []

    for lam in lambda_list:

        selector = BanditSelector(
            prior_alpha = np.ones(p),
            prior_beta = np.ones(p),
            pi_thr = cfg.pi_thr,
            rng = rng
        )

        learner = LassoLearner(
            lam = lam,
            rng = rng
        )

        runner = LassoBanditRunner(
            selector = selector,
            learner = learner
        )

        runner.run(X, y, cfg.num_iter)

        print(f"Finished running LASSO bandit, the selected set is: \n {runner.stable_set_}")

        results = {
            "selection_prob": runner.selection_prob_,
            "selected_sets": runner.selected_sets_
        }

        result_list.append(results)


    # Save results
    run_dir = make_run_dir(Path("results"), tag="lasso_bandit")

    save_results(run_dir, [lambda_list, result_list])
    save_config(run_dir, cfg)

    print(f"Results saved to {run_dir}.")

if __name__ == "__main__":
    main()


