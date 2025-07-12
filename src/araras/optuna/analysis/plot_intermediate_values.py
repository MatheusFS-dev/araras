"""
Module plot_intermediate_values of analysis

Functions:
    - plot_intermediate_values: Plot intermediate values reported during trials.

Example:
    >>> from araras.optuna.analysis.plot_intermediate_values import plot_intermediate_values
    >>> plot_intermediate_values(...)
"""
from araras.commons import *
import os
import matplotlib.pyplot as plt
import optuna
import numpy as np

from .analyzer import (
    PLOT_CFG,
    draw_warning_box,
    save_plot,
    save_plotly_html,
)


def plot_intermediate_values(
    study: optuna.Study, dirs: Dict[str, str], create_plotly: bool = False
) -> None:
    """Plot intermediate values reported during trials.

    Parameters
    ----------
    create_plotly : bool
        Whether to save an interactive HTML version of the plot.
    """
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for intermediate values plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "study_intermediate_values", "figs", create_plotly)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
    cmap = plt.get_cmap("tab10")
    for idx, t in enumerate(trials):
        if not t.intermediate_values:
            continue
        steps = sorted(t.intermediate_values.keys())
        values = [t.intermediate_values[s] for s in steps]
        color = cmap(idx % 10)
        ax.plot(
            steps,
            values,
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Step", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Intermediate Values", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_intermediate_values(study)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "study_intermediate_values", "figs", create_plotly, pfig)
    plt.close(fig)
