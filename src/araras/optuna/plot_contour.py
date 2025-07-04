import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import numpy as np
from itertools import combinations
from matplotlib.tri import Triangulation

from .analyze import (
    PLOT_CFG,
    format_title,
    get_param_display_name,
)


def plot_contour(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
) -> None:
    """Generate contour plots for parameter pairs.

    This creates a single multipanel figure covering all provided parameters
    and optionally standalone figures for each pair of parameters.
    """
    if not params:
        print("No parameters available for contour plot.")
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={'value': 'loss'})

    pairs = list(combinations(params, 2))
    if not pairs:
        print("Need at least two parameters for contour plot.")
        return

    max_cols = PLOT_CFG.max_cols
    n_plots = len(pairs)
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + max_cols - 1) // max_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(PLOT_CFG.numeric_subplot_size * n_cols, PLOT_CFG.numeric_subplot_size * n_rows),
    )

    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (p1, p2) in enumerate(pairs):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row, col]

        x = df[p1].to_numpy()
        y = df[p2].to_numpy()
        z = df["loss"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        else:
            tri = Triangulation(x, y)
            cf = ax.tricontourf(tri, z, levels=20, cmap="Blues")
            ax.scatter(x, y, c=z, cmap="Blues", s=15, edgecolor="black", linewidth=0.2)
            fig.colorbar(cf, ax=ax, label=PLOT_CFG.study_value_label)

        ax.set_xlabel(get_param_display_name(p1), fontsize=PLOT_CFG.label_fs)
        ax.set_ylabel(get_param_display_name(p2), fontsize=PLOT_CFG.label_fs)
        title = f"{get_param_display_name(p1)} vs {get_param_display_name(p2)}"
        ax.set_title(
            format_title(PLOT_CFG.param_title_tpl, title),
            fontsize=PLOT_CFG.title_fs,
            pad=PLOT_CFG.title_pad,
        )
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "params_contour.pdf"), bbox_inches="tight")
    plt.close(fig)

    if create_standalone:
        for p1, p2 in pairs:
            fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

            x = df[p1].to_numpy()
            y = df[p2].to_numpy()
            z = df["loss"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[mask], y[mask], z[mask]

            if len(x) >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
                tri = Triangulation(x, y)
                cf = ax.tricontourf(tri, z, levels=20, cmap="Blues")
                ax.scatter(x, y, c=z, cmap="Blues", s=20, edgecolor="black", linewidth=0.2)
                fig.colorbar(cf, ax=ax, label=PLOT_CFG.study_value_label)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    ha="center",
                    va="center",
                    fontsize=PLOT_CFG.standalone_label_fs,
                )

            ax.set_xlabel(get_param_display_name(p1), fontsize=PLOT_CFG.standalone_label_fs)
            ax.set_ylabel(get_param_display_name(p2), fontsize=PLOT_CFG.standalone_label_fs)
            title = f"{get_param_display_name(p1)} vs {get_param_display_name(p2)}"
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, title),
                fontsize=PLOT_CFG.standalone_title_fs,
                pad=PLOT_CFG.standalone_title_pad,
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"contour_{p1}_{p2}.pdf"
            fig.savefig(os.path.join(dirs["standalone_contours"], filename), bbox_inches="tight")
            plt.close(fig)
