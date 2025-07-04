import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import numpy as np
from itertools import combinations
import pandas as pd

from .analyze import (
    PLOT_CFG,
    get_param_display_name,
    format_title,
)


def plot_rank(study: optuna.Study, params: List[str], dirs: Dict[str, str]) -> None:
    """Plot parameter relations colored by rank."""
    if not params:
        print("No parameters available for rank plot.")
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={"value": "loss"})
    if df.empty:
        print("No completed trials for rank plot.")
        return

    # Separate numeric and categorical parameters
    numeric_params = []
    categorical_params = []

    for param in params:
        if param in df.columns:
            # Check if parameter values are numeric
            try:
                pd.to_numeric(df[param].dropna())
                numeric_params.append(param)
            except (ValueError, TypeError):
                categorical_params.append(param)

    # Only create pairs from numeric parameters for scatter plots
    pairs = list(combinations(numeric_params, 2))
    if not pairs:
        print("Need at least two numeric parameters for rank plot.")
        return

    loss_vals = df["loss"].replace([np.inf, -np.inf], np.nan)
    cmap = plt.cm.coolwarm
    vmin = loss_vals.min()
    vmax = loss_vals.max()

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

        # Ensure data is numeric
        x_data = pd.to_numeric(df[p1], errors="coerce")
        y_data = pd.to_numeric(df[p2], errors="coerce")

        # Remove NaN values
        valid_mask = ~(x_data.isna() | y_data.isna())
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        c_data = loss_vals[valid_mask]

        sc = ax.scatter(
            x_data,
            y_data,
            c=c_data,
            cmap=cmap,
            s=20,
            edgecolor="black",
            linewidth=0.2,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(get_param_display_name(p1), fontsize=PLOT_CFG.label_fs)
        ax.set_ylabel(get_param_display_name(p2), fontsize=PLOT_CFG.label_fs)
        title = f"{get_param_display_name(p1)} vs {get_param_display_name(p2)}"
        ax.set_title(
            format_title(PLOT_CFG.param_title_tpl, title),
            fontsize=PLOT_CFG.title_fs,
            pad=PLOT_CFG.title_pad,
        )
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label=PLOT_CFG.study_value_label)

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_rank.pdf"), bbox_inches="tight")
    plt.close(fig)
