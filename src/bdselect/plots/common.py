
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt


def new_fig(figsize = (7, 5)):
    fig, ax = plt.subplots(figsize = figsize)
    return fig, ax


def save_fig(fig, path: str | Path, dpi: int = 200):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(path, dpi = dpi, bbox_inches = "tight")


def style_axes(ax, title, y_label, x_label):
    ax.grid(True, which = "both", axis = "y", linestyle = "--", alpha = 0.4)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend()


