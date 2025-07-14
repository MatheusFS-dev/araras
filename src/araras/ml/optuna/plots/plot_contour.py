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
from itertools import combinations
from matplotlib.tri import Triangulation

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    format_title,
    get_param_display_name,
    calculate_grid,
    draw_warning_box,
    save_plot,
)


def plot_contour(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
    create_plotly: bool = False,
) -> None:
    """Generate contour plots for parameter pairs.

    This creates a single multipanel figure covering all provided parameters
    and optionally standalone figures for each pair of parameters.

    Parameters
    ----------
    create_plotly : bool
        Whether to save interactive HTML versions of the plots.
    """
    if not params:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No parameters available for contour plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_contour", "figs", create_plotly)
        plt.close(fig)
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={"value": "loss"})
    if df.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for contour plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_contour", "figs", create_plotly)
        plt.close(fig)
        return

    pairs = list(combinations(params, 2))
    if not pairs:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "Need at least two parameters for contour plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_contour", "figs", create_plotly)
        plt.close(fig)
        return

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

        x = df[p1].to_numpy()
        y = df[p2].to_numpy()
        z = df["loss"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[mask], y[mask], z[mask]

        unique_points = np.unique(np.column_stack((x, y)), axis=0)
        if unique_points.shape[0] < 3:
            draw_warning_box(ax, "No Data or Error Generating")
        else:
            try:
                tri = Triangulation(x, y)
                cf = ax.tricontourf(tri, z, levels=20, cmap="coolwarm")
                ax.scatter(
                    x,
                    y,
                    c=z,
                    cmap="coolwarm",
                    s=15,
                    edgecolor="black",
                    linewidth=0.2,
                )
                fig.colorbar(cf, ax=ax, label=PLOT_CFG.study_value_label)
            except Exception as e:
                logger.warning(f"{YELLOW}Error generating contour for {p1} vs {p2}: {e}{RESET}")
                draw_warning_box(ax, f"{e}")

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
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_contour(study, params=params)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "params_contour", "figs", create_plotly, pfig)
    plt.close(fig)

    if create_standalone:
        for p1, p2 in pairs:
            fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

            x = df[p1].to_numpy()
            y = df[p2].to_numpy()
            z = df["loss"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[mask], y[mask], z[mask]

            unique_points = np.unique(np.column_stack((x, y)), axis=0)
            if unique_points.shape[0] < 3:
                ax.text(0.5, 0.5, draw_warning_box(ax, "No Data or Error Generating"))
            else:
                try:
                    tri = Triangulation(x, y)
                    cf = ax.tricontourf(tri, z, levels=20, cmap="coolwarm")
                    ax.scatter(
                        x,
                        y,
                        c=z,
                        cmap="coolwarm",
                        s=20,
                        edgecolor="black",
                        linewidth=0.2,
                    )
                    fig.colorbar(cf, ax=ax, label=PLOT_CFG.study_value_label)
                except Exception as e:
                    logger.warning(
                        f"{YELLOW}Error generating standalone contour for {p1} vs {p2}: {e}{RESET}"
                    )
                    draw_warning_box(ax, f"{e}")

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
            filename = f"contour_{p1}_{p2}"
            pfig_s = None
            if create_plotly:
                try:
                    pfig_s = ov.plot_contour(study, params=[p1, p2])
                except Exception:
                    pfig_s = None

            save_plot(fig, dirs, filename, "standalone_contours", create_plotly, pfig_s)
            plt.close(fig)
