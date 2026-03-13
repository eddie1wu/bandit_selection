
from pathlib import Path
import matplotlib.pyplot as plt

from datetime import datetime

from bdselect.io.save_load import load_results
from bdselect.plots.common import new_fig, save_fig, style_axes


def plot_prob_over_iter():

    # Define path
    res_paths = [
        Path("results/2026-03-09_193712_stability_selection"),
        Path("results/2026-03-09_193720_lasso_bandit")
    ]

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Load results
    dim_signal = 10
    dim_X = 1000

    ss_results = load_results(res_paths[0])
    bandit_results = load_results(res_paths[1])

    ss_results = ss_results[1]
    ss_selection_prob = ss_results[0]["selection_prob"]
    ss_selection_prob = list(zip(*ss_selection_prob))

    bandit_results = bandit_results[1]
    bandit_selection_prob = bandit_results[0]["selection_prob"]
    bandit_selection_prob = list(zip(*bandit_selection_prob))


    # Plotting function
    def _make_plot(selection_prob, title, y_label, x_label, out_path):
        fig, ax = new_fig()
        x_data = range(len(selection_prob[0]))

        for i in range(dim_X):
            if i < dim_signal:
                ax.plot(x_data, selection_prob[i], color = "red", alpha = 1, lw = 1.2)
            else:
                ax.plot(x_data, selection_prob[i], color = "black", alpha = 0.4, lw = 0.6)

        style_axes(ax, title, y_label, x_label)

        # Save fig
        save_fig(fig, out_path)

        print(f"Graph saved to {out_path}.")


    # Make plots
    _make_plot(
        ss_selection_prob,
        "Stability Selection Probabilities Over Time (Orthogonal X)",
        "Selection probability",
        "Iteration",
        Path(f"graphs/{ts}_ss_probability_over_iter_lambda5.png")
    )

    _make_plot(
        bandit_selection_prob,
        "Bandit Selection Probabilities Over Time (Orthogonal X)",
        "Selection probability",
        "Iteration",
        Path(f"graphs/{ts}_bandit_probability_over_iter_lambda5.png")
    )


if __name__ == '__main__':
    plot_prob_over_iter()
