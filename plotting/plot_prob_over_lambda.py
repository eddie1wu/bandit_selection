
from pathlib import Path
from select import select

import matplotlib.pyplot as plt

from datetime import datetime

from bdselect.io.save_load import load_results
from bdselect.metrics import compute_roc
from bdselect.plots.common import new_fig, save_fig, style_axes


def plot_prob_over_lambda():

    # Define path
    res_paths = [
        Path("results/2026-03-10_103443_stability_selection"),
        Path("results/2026-03-10_104044_lasso_bandit")
    ]

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Load results
    dim_signal = 10
    dim_X = 1000

    ss_results = load_results(res_paths[0])
    # lambda_list = ss_results[0]
    ss_prob_list = ss_results[1]
    ss_prob_list = [x["selection_prob"][-1] for x in ss_prob_list]
    ss_prob_list = list(zip(*ss_prob_list))

    bandit_results = load_results(res_paths[1])
    lambda_list = bandit_results[0]
    bandit_prob_list = bandit_results[1]
    bandit_prob_list = [x["selection_prob"][-1] for x in bandit_prob_list]
    bandit_prob_list = list(zip(*bandit_prob_list))


    # Plotting function
    def _make_plot(selection_prob, lambda_grid, title, y_label, x_label, out_path):

        fig, ax = new_fig()

        x_data = range(len(selection_prob[0]))

        for i in range(len(selection_prob)-1, -1, -1):
            if i < dim_signal:
                ax.plot(lambda_grid, selection_prob[i], color = "red", alpha = 1, lw = 1.2)
            else:
                ax.plot(lambda_grid, selection_prob[i], color = "black", alpha = 0.4, lw = 0.6)

        ax.invert_xaxis()
        ax.set_xscale("log")
        style_axes(ax, title, y_label, x_label)


        # Save fig
        save_fig(fig, out_path)

        print(f"Graph saved to {out_path}.")

    # Make plot
    _make_plot(
        ss_prob_list,
        lambda_list,
        "Stability Selection Probability against Lambda (Orthogonal X)",
        "Selection probability",
        "Lambda",
        Path(f"graphs/{ts}_ss_probability_over_lambda.png")
    )

    _make_plot(
        bandit_prob_list,
        lambda_list,
        "Bandit Selection Probability against Lambda (Orthogonal X)",
        "Selection probability",
        "Lambda",
        Path(f"graphs/{ts}_bandit_probability_over_lambda.png")
    )



if __name__ == "__main__":
    plot_prob_over_lambda()

