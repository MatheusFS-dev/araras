import os
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib as mpl
import optuna
import numpy as np
from itertools import combinations
import pandas as pd

from .analyze import (
    PLOT_CFG,
    get_param_display_name,
    format_title,
    calculate_grid,
    draw_warning_box,
)


def plot_rank(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
) -> None:
    """Plot parameter relations colored by rank."""
    if not params:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No parameters available for rank plot.")
        plt.tight_layout()
        fig.savefig(os.path.join(dirs["figs"], "params_study_value_rank.pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={"value": "loss"})
    if df.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for rank plot.")
        plt.tight_layout()
        fig.savefig(os.path.join(dirs["figs"], "params_study_value_rank.pdf"), bbox_inches="tight")
        plt.close(fig)
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
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "Need at least two numeric parameters for rank plot.")
        plt.tight_layout()
        fig.savefig(os.path.join(dirs["figs"], "params_study_value_rank.pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    loss_vals = df["loss"].replace([np.inf, -np.inf], np.nan)
    cmap = mpl.colormaps['coolwarm']
    vmin = loss_vals.min()
    vmax = loss_vals.max()

    max_cols = PLOT_CFG.max_cols + 2
    n_plots = len(pairs)
    n_rows, n_cols = calculate_grid(
        n_plots,
        PLOT_CFG.numeric_subplot_size,
        PLOT_CFG.numeric_subplot_size,
        max_cols,
    )

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
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        try:
            x_data = pd.to_numeric(df[p1], errors="coerce")
            y_data = pd.to_numeric(df[p2], errors="coerce")

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
        except Exception as e:
            draw_warning_box(ax, f"Could not plot pair: {e}")
            continue
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
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "params_study_value_rank.pdf"), bbox_inches="tight")
    plt.close(fig)

    if create_standalone:
        for p1, p2 in pairs:
            fig_s, ax_s = plt.subplots(figsize=PLOT_CFG.standalone_size)

            x_data = pd.to_numeric(df[p1], errors="coerce")
            y_data = pd.to_numeric(df[p2], errors="coerce")
            valid_mask = ~(x_data.isna() | y_data.isna())
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            c_data = loss_vals[valid_mask]

            sc = ax_s.scatter(
                x_data,
                y_data,
                c=c_data,
                cmap=cmap,
                s=25,
                edgecolor="black",
                linewidth=0.3,
                vmin=vmin,
                vmax=vmax,
            )
            ax_s.set_xlabel(get_param_display_name(p1), fontsize=PLOT_CFG.standalone_label_fs)
            ax_s.set_ylabel(get_param_display_name(p2), fontsize=PLOT_CFG.standalone_label_fs)
            title = f"{get_param_display_name(p1)} vs {get_param_display_name(p2)}"
            ax_s.set_title(
                format_title(PLOT_CFG.param_title_tpl, title),
                fontsize=PLOT_CFG.standalone_title_fs,
                pad=PLOT_CFG.standalone_title_pad,
            )
            ax_s.grid(True, alpha=0.3)
            fig_s.colorbar(sc, ax=ax_s, label=PLOT_CFG.study_value_label)
            ax_s.tick_params(axis="x", labelsize=PLOT_CFG.standalone_x_tick_fs)
            ax_s.tick_params(axis="y", labelsize=PLOT_CFG.standalone_y_tick_fs)

            plt.tight_layout()
            fname = os.path.join(dirs["standalone_ranks"], f"rank_{p1}_{p2}.pdf")
            fig_s.savefig(fname, bbox_inches="tight")
            plt.close(fig_s)
