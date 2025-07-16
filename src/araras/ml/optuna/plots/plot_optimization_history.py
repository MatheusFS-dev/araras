"""
Last Edited: 14 July 2025
Description:
    Visualize optimization history.
"""
from araras.core import *

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from araras.ml.optuna.analyzer import PLOT_CFG, draw_warning_box, save_plot, save_plotly_html


def plot_optimization_history(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool = False) -> None:
    """Plot optimization history of the study.

    This chart displays how the objective value evolves with each trial. Use it
    to diagnose convergence trends. When ``create_plotly`` is ``True`` an
    interactive version of the plot is also generated.

    Args:
        study: Optuna study to visualize.
        dirs: Dictionary with output directories for saving figures.
        create_plotly: Whether to save an interactive HTML version of the plot.

    Returns:
        None

    Raises:
        None
    """
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df = df.query("state == 'COMPLETE'")
    if df.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for optimization history plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "study_optimization_history", "figs", create_plotly)
        plt.close(fig)
        return

    df = df.rename(columns={"value": "loss"})
    df = df.sort_values("number")
    loss = df["loss"].replace([float("inf"), float("-inf")], pd.NA)

    best = loss.cummin()

    fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
    ax.scatter(df["number"], loss, color="blue", edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.plot(df["number"], best, color="red", linewidth=2, label="Best so far")

    ax.set_xlabel("Trial", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Optimization History", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    ax.legend(fontsize=PLOT_CFG.legend_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_optimization_history(study)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "study_optimization_history", "figs", create_plotly, pfig)
    plt.close(fig)
