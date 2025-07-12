"""
Module plot_optimization_history of analysis

Functions:
    - plot_optimization_history: Plot optimization history of the study.

Example:
    >>> from araras.optuna.analysis.plot_optimization_history import plot_optimization_history
    >>> plot_optimization_history(...)
"""
from araras.commons import *
import os
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import plotly.io as pio

from .analyzer import PLOT_CFG, draw_warning_box, save_plotly_html


def plot_optimization_history(
    study: optuna.Study,
    dirs: Dict[str, str],
    save_plotly: bool = False,
) -> None:
    """Plot optimization history of the study."""
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df = df.query("state == 'COMPLETE'")
    if df.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for optimization history plot.")
        plt.tight_layout()
        fig.savefig(os.path.join(dirs["figs"], "study_optimization_history.pdf"), bbox_inches="tight")
        if save_plotly and dirs.get("plotly"):
            save_plotly_html(fig, os.path.join(dirs["plotly"], "study_optimization_history.html"))
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
    fig.savefig(os.path.join(dirs["figs"], "study_optimization_history.pdf"), bbox_inches="tight")
    if save_plotly and dirs.get("plotly"):
        save_plotly_html(fig, os.path.join(dirs["plotly"], "study_optimization_history.html"))
    plt.close(fig)
