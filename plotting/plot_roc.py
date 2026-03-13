
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

from datetime import datetime

from bdselect.io.save_load import load_results
from bdselect.metrics import compute_roc
from bdselect.plots.common import new_fig, save_fig, style_axes


def plot_roc():

    # Define path
    res_paths = [
        Path("results/2026-03-08_222505_stability_selection"),
        Path("results/2026-03-08_231550_lasso_bandit")
    ]

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Load results
    dim_signal = 10
    dim_X = 1000

    ss_results = load_results(res_paths[0])
    ss_prob_list = ss_results[1]
    bandit_results = load_results(res_paths[1])
    bandit_prob_list = bandit_results[1]

    def get_tpr_fpr_list(selection_prob_list):

        # Compute TPR FPR for each lambda value
        tpr_list = []
        fpr_list = []

        for j in range(len(selection_prob_list)):

            selection_prob = selection_prob_list[j]["selection_prob"][-1]
            stable_set = np.argwhere(selection_prob >= 0.5).reshape(-1)

            true_set = set(range(dim_signal))
            tpr, fpr, accuracy = compute_roc(stable_set, true_set, dim_X)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))

        return tpr_list, fpr_list


    # Plot ROC
    ss_tpr_list, ss_fpr_list = get_tpr_fpr_list(ss_prob_list)
    bandit_tpr_list, bandit_fpr_list = get_tpr_fpr_list(bandit_prob_list)

    fig, ax = new_fig()
    # ax.plot(ss_fpr_list, ss_tpr_list, label='Stability selection', color='C0')
    ax.plot(bandit_fpr_list, bandit_tpr_list, label='Bandit selection', color="#ff7f0e")
    style_axes(ax, "ROC Curves of Stability Selection vs Bandit Selection", "True positive rate", "False positive rate")

    # Save fig
    out_path = Path(f"graphs/{ts}_roc_curve.png")
    save_fig(fig, out_path)

    print(f"Graph saved to {out_path}")



if __name__ == "__main__":
    plot_roc()



