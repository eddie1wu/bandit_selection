import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bdselect.models.BanditSelector import BanditSelector
from bdselect.models.LassoLearnerCV import LassoLearnerCV
from bdselect.models.LassoCVBanditRunner import LassoBanditRunner
from bdselect.rng import make_rng
from bdselect.io.save_load import load_config
from bdselect.gen_data import gen_factor_data, gen_orthog_data


def main():

    # Load config
    config_path = Path("configs/lasso_bandit.toml")
    cfg = load_config(config_path)
    rng = make_rng(cfg.seed)

    # Generate data
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

    y = np.asarray(y).reshape(-1)
    n, p = X.shape

    lambda_grid = np.logspace(-4, 2, 200)

    one_se = True

    selector = BanditSelector(
        prior_alpha=np.ones(p),
        prior_beta=np.ones(p),
        pi_thr=cfg.pi_thr,
        rng=rng
    )

    learner = LassoLearnerCV(
        lambda_grid=lambda_grid,
        rng=rng,
        cv=5,
        use_1se=one_se
    )

    runner = LassoBanditRunner(
        selector=selector,
        learner=learner
    )

    runner.run(X, y, T=250)

    print(f"Finished running LASSO bandit, stable set is:\n{runner.stable_set_}")

    selection_prob = np.asarray(runner.selection_prob_)   # shape (T, p)
    chosen_lambda = np.asarray(runner.chosen_lambda_, dtype=float)

    true_idx = np.arange(cfg.dim_signal)
    noise_idx = np.arange(cfg.dim_signal, p)

    x = np.arange(1, selection_prob.shape[0] + 1)

    # Plot posterior means
    plt.figure(figsize=(7, 5))

    for j in noise_idx:
        plt.plot(x, selection_prob[:, j], color="black", alpha=0.4, linewidth=0.6)

    for j in true_idx:
        plt.plot(x, selection_prob[:, j], color="red", alpha=1, linewidth=1.2)

    # plt.axhline(cfg.pi_thr, linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Selection probability")
    plt.title("Bandit Selection Probabilities with Varying Penalty (Orthogonal X)")
    plt.tight_layout()
    plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    plt.savefig(f"graphs/bandit_probability_over_iter_vary_lambda_{one_se}.png", dpi = 200, bbox_inches = "tight")

    # plt.show()

    # Plot chosen lambda over iterations
    plt.figure(figsize=(7, 5))
    plt.plot(x, chosen_lambda, linewidth=1.2)
    plt.xlabel("Iteration")
    plt.ylabel("Chosen lambda")

    if one_se:
        title = "CV-chosen lambda (+1 s.e.) over iterations"
    else:
        title = "CV-chosen lambda over iterations"
    plt.title(title)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"graphs/bandit_lambda_over_iter_{one_se}.png", dpi=200, bbox_inches="tight")
    # plt.show()




if __name__ == "__main__":
    main()


