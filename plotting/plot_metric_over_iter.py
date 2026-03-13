
from pathlib import Path
import matplotlib.pyplot as plt

from datetime import datetime

from bdselect.io.save_load import load_results
from bdselect.metrics import compute_roc
from bdselect.plots.common import new_fig, save_fig, style_axes


def plot_metric_over_iter():

    # Define path
    res_paths = [       # lambda = 1e1
        Path("results/2026-03-08_195238_stability_selection"),
        Path("results/2026-03-08_195242_lasso_bandit")
    ]

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")


    # Load results
    dim_signal = 10
    dim_X = 1000

    ss_results = load_results(res_paths[0])
    ss_selected_sets = ss_results["selected_sets"]

    bandit_results = load_results(res_paths[1])
    bandit_selected_sets = bandit_results["selected_sets"]

    # Compute metrics
    ss_accuracy = []
    ss_TPR = []
    ss_FPR = []
    bandit_accuracy = []
    bandit_TPR = []
    bandit_FPR = []

    true_set = set(range(dim_signal))

    for t in range(len(ss_selected_sets)):
        TPR, FPR, acc = compute_roc(ss_selected_sets[t], true_set, dim_X)
        ss_accuracy.append(acc)
        ss_TPR.append(TPR)
        ss_FPR.append(FPR)
        TPR, FPR, acc = compute_roc(bandit_selected_sets[t], true_set, dim_X)
        bandit_accuracy.append(acc)
        bandit_TPR.append(TPR)
        bandit_FPR.append(FPR)


    # Plotting function
    def _make_plot(ss_data, bandit_data, title, y_label, x_label, out_path):
        fig, ax = new_fig()
        x_data = range(len(ss_data))

        ax.plot(x_data, ss_data, label='stability selection', color = 'C0')
        ax.plot(x_data, bandit_data, label='bandit selection', color = '#ff7f0e')
        style_axes(ax, title, y_label, x_label)

        # Save fig
        save_fig(fig, out_path)

        print(f"Graph saved to {out_path}.")

    # Make plot
    _make_plot(
        ss_accuracy,
        bandit_accuracy,
        "Comparing Accuracy of Stability Selection vs Bandit",
        "Selection accuracy",
        "Iteration",
        Path(f"graphs/{ts}_acc_over_iter.png")
    )

    _make_plot(
        ss_TPR,
        bandit_TPR,
        "Comparing TPR of Stability Selection vs Bandit",
        "True positive rate",
        "Iteration",
        Path(f"graphs/{ts}_tpr_over_iter.png")
    )

    _make_plot(
        ss_FPR,
        bandit_FPR,
        "Comparing FPR of Stability Selection vs Bandit",
        "False positive rate",
        "Iteration",
        Path(f"graphs/{ts}_fpr_over_iter.png")
    )


if __name__ == "__main__":
    plot_metric_over_iter()


