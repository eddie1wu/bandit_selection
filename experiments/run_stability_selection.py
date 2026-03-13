import numpy as np

from pathlib import Path

from tqdm import tqdm

from bdselect.gen_data import gen_factor_data, gen_orthog_data
from bdselect.models.StabilitySelection import StabilitySelection
from bdselect.rng import make_rng
from bdselect.io.save_load import load_config, save_config, make_run_dir, save_results


def main():

    # Load config
    config_path = Path("configs/stability_selection.toml")
    cfg = load_config(config_path)
    rng = make_rng(cfg.seed)

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

    lambda_list = np.logspace(-1, 2, 50)
    # lambda_list = [5]
    result_list = []

    for lam in lambda_list:

        # Run stability selection for a fixed lambda
        stable_select = StabilitySelection(
            lam = lam,
            num_iter = cfg.num_iter,
            subsample_fraction = cfg.subsample_fraction,
            weakness = cfg.weakness,
            pw = cfg.pw,
            pi_thr = cfg.pi_thr,
            rng = rng
        )

        stable_select.fit(X, y)

        print(f"Finished stability selection for lambda = {lam}, the selected variables are: \n {stable_select.stable_set_}")

        results = {
            "selection_prob": stable_select.selection_prob_,
            "selected_sets": stable_select.selected_sets_
        }

        result_list.append(results)

    # Save results
    run_dir = make_run_dir(Path("results"), tag="stability_selection")

    save_results(run_dir, [lambda_list, result_list])
    save_config(run_dir, cfg)

    print(f"Results saved to {run_dir}.")


if __name__ == '__main__':
    main()


