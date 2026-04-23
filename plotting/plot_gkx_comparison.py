import matplotlib.pyplot as plt


def plot_rank_comparison(
    list1,
    list2,
    top_k=None,
    figsize=(8, 10),
    title="Rank Comparison (Slope Chart)"
):
    """
    Plot a slope chart comparing rankings from two lists of strings.

    Parameters
    ----------
    list1 : list of str
        Ranking from model 1 (index 0 = most important)

    list2 : list of str
        Ranking from model 2

    top_k : int or None
        If not None, only plot top_k variables from each list

    """

    # Optionally truncate
    if top_k is not None:
        list1 = list1[:top_k]
        list2 = list2[:top_k]

    # Rank dictionaries
    rank1 = {v: i for i, v in enumerate(list1)}
    rank2 = {v: i for i, v in enumerate(list2)}

    # Union of variables
    all_vars = sorted(set(list1) | set(list2))

    # Create figure
    plt.figure(figsize=figsize)

    # Define colors
    solid_color = "#1f77b4"
    dashed_color = "#999999"

    common_vars = set(rank1) & set(rank2)

    # --- Step 1: plot solid lines ---
    rows_with_solid_left = set()
    rows_with_solid_right = set()

    for v in common_vars:
        y1 = rank1[v]
        y2 = rank2[v]

        plt.plot([0, 1], [y1, y2],
                 color=solid_color,
                 alpha=0.7,
                 linewidth=2)

        rows_with_solid_left.add(y1)
        rows_with_solid_right.add(y2)

    # --- Step 2: add dashed lines ONLY if both sides have no solid at that row ---
    max_len = max(len(list1), len(list2))

    for i in range(max_len):
        if (i not in rows_with_solid_left) and (i not in rows_with_solid_right):
            plt.plot([0, 1], [i, i],
                     linestyle="dashed",
                     color=dashed_color,
                     alpha=0.5,
                     linewidth=1)

    # Left labels
    for v, y in rank1.items():
        plt.text(-0.05, y, v, ha='right', va='center')

    # Right labels
    for v, y in rank2.items():
        plt.text(1.05, y, v, ha='left', va='center')

    # Formatting
    plt.gca().invert_yaxis()  # rank 0 at top
    plt.xticks([0, 1], ["Model 1", "Model 2"])
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"graphs/gkx_rank_comparison.png", dpi=200, bbox_inches="tight")




my_features = [
    "indmom", "baspread", "mom12m", "beta", "mom1m",
    "idiovol", "mom36m", "mom6m", "herf", "turn",
    "mvel1", "rd_sale", "dolvol", "retvol", "maxret",
    "cash", "std_dolvol", "betasq", "bm_x", "ep_x",
    "roaq", "mve_ia", "chmom", "orgcap", "rd_mve", "depr"
]

gkx_features = [
    "mom1m", "mvel1", "mom12m", "chmom", "maxret",
    "indmom", "retvol", "dolvol", "sp", "turn",
    "agr", "nincr", "rd_mve", "std_turn", "mom6m",
    "mom36m", "chcsho", "securedind", "idiovol", "baspread",
    "ill", "age", "convind", "rd", "depr",
    "beta", "betasq"
]

plot_rank_comparison(my_features, gkx_features, title="Feature Rank Comparison")


