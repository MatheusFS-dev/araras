import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd

from .analyze import (
    PLOT_CFG,
    get_param_display_name,
    format_title,
)


def plot_slice(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
) -> None:
    """Create slice plots for each parameter."""
    if not params:
        print("No parameters available for slice plot.")
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={'value': 'loss'})
    if df.empty:
        print("No completed trials for slice plot.")
        return

    max_cols = PLOT_CFG.max_cols
    n_plots = len(params)
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

    for idx, p in enumerate(params):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row, col]

        x = df[p].to_numpy()
        y = df["loss"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        ax.scatter(x, y, s=20, edgecolor="black", linewidth=0.2, alpha=0.6)

        if len(x) > 1:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            window = max(2, len(x_sorted) // 20)
            smooth = pd.Series(y_sorted).rolling(window=window, min_periods=1, center=True).mean()
            ax.plot(x_sorted, smooth, color="red", linewidth=2)

        ax.set_xlabel(get_param_display_name(p), fontsize=PLOT_CFG.label_fs)
        ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.label_fs)
        ax.set_title(
            format_title(PLOT_CFG.param_title_tpl, get_param_display_name(p)),
            fontsize=PLOT_CFG.title_fs,
            pad=PLOT_CFG.title_pad,
        )
        ax.grid(True, alpha=0.3)

        if create_standalone:
            fig_s, ax_s = plt.subplots(figsize=PLOT_CFG.standalone_size)
            ax_s.scatter(x, y, s=20, edgecolor="black", linewidth=0.2, alpha=0.6)
            if len(x) > 1:
                ax_s.plot(x_sorted, smooth, color="red", linewidth=2)
            ax_s.set_xlabel(get_param_display_name(p), fontsize=PLOT_CFG.standalone_label_fs)
            ax_s.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
            ax_s.set_title(
                format_title(PLOT_CFG.param_title_tpl, get_param_display_name(p)),
                fontsize=PLOT_CFG.standalone_title_fs,
                pad=PLOT_CFG.standalone_title_pad,
            )
            ax_s.grid(True, alpha=0.3)
            ax_s.tick_params(axis="x", labelsize=PLOT_CFG.standalone_x_tick_fs)
            ax_s.tick_params(axis="y", labelsize=PLOT_CFG.standalone_y_tick_fs)
            plt.tight_layout()
            fig_s.savefig(os.path.join(dirs["standalone_slices"], f"slice_{p}.pdf"), bbox_inches="tight")
            plt.close(fig_s)

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "params_slice.pdf"), bbox_inches="tight")
    plt.close(fig)
