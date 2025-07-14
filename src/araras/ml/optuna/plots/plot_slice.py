"""
Last Edited: 14 July 2025
Description:
    This script demonstrates a detailed 
    header format with additional metadata.
"""
from araras.core import *

import matplotlib.pyplot as plt
import optuna
import numpy as np
import pandas as pd

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    get_param_display_name,
    format_title,
    calculate_grid,
    draw_warning_box,
    save_plot,
)


def plot_slice(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
    create_plotly: bool = False,
) -> None:
    """Create slice plots for each parameter.

    Parameters
    ----------
    create_plotly : bool
        Whether to save interactive HTML versions of the plots.
    """
    if not params:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No parameters available for slice plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_slice", "figs", create_plotly)
        plt.close(fig)
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={"value": "loss"})
    if df.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for slice plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_slice", "figs", create_plotly)
        plt.close(fig)
        return

    # Filter to only numeric parameters for slice plots
    numeric_params = []
    for param in params:
        if param in df.columns:
            try:
                pd.to_numeric(df[param].dropna(), errors="raise")
                numeric_params.append(param)
            except (ValueError, TypeError):
                continue  # Skip categorical parameters

    if not numeric_params:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No numeric parameters available for slice plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_slice", "figs", create_plotly)
        plt.close(fig)
        return

    max_cols = PLOT_CFG.max_cols
    n_plots = len(numeric_params)
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

    for idx, p in enumerate(numeric_params):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        try:
            x = pd.to_numeric(df[p], errors="coerce").to_numpy()
            y = df["loss"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) == 0:
                draw_warning_box(ax, f"No valid data for {get_param_display_name(p)}")
                ax.set_title(
                    format_title(PLOT_CFG.param_title_tpl, get_param_display_name(p)),
                    fontsize=PLOT_CFG.title_fs,
                    pad=PLOT_CFG.title_pad,
                )
                continue

            ax.scatter(x, y, s=20, edgecolor="black", linewidth=0.2, alpha=0.6)

            if len(x) > 1:
                order = np.argsort(x)
                x_sorted = x[order]
                y_sorted = y[order]
                window = max(2, len(x_sorted) // 20)
                smooth = pd.Series(y_sorted).rolling(window=window, min_periods=1, center=True).mean()
                ax.plot(x_sorted, smooth, color="red", linewidth=2)
        except Exception as e:
            draw_warning_box(ax, f"Error plotting {get_param_display_name(p)}: {e}")
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, get_param_display_name(p)),
                fontsize=PLOT_CFG.title_fs,
                pad=PLOT_CFG.title_pad,
            )
            continue

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
            pfig_s = None
            if create_plotly:
                import optuna.visualization as ov

                try:
                    pfig_s = ov.plot_slice(study, params=[p])
                except Exception:
                    pfig_s = None

            save_plot(fig_s, dirs, f"slice_{p}", "standalone_slices", create_plotly, pfig_s)
            plt.close(fig_s)

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_slice(study, params=numeric_params)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "params_slice", "figs", create_plotly, pfig)
    plt.close(fig)
