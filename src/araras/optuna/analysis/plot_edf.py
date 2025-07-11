"""
Module plot_edf of analysis

Functions:
    - plot_edf: Plot the empirical distribution function of objective values.

Example:
    >>> from araras.optuna.analysis.plot_edf import plot_edf
    >>> plot_edf(...)
"""
from araras.commons import *
import os
import matplotlib.pyplot as plt
import optuna
import numpy as np
import scipy.interpolate

from .analyzer import (
    PLOT_CFG,
    draw_warning_box,
)


def plot_edf(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Plot the empirical distribution function of objective values."""
    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'")
    values = df["value"].astype(float).sort_values()
    if values.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for EDF plot.")
        ax.set_title(
            "Empirical Distribution of Study Values",
            pad=PLOT_CFG.title_pad,
            fontsize=PLOT_CFG.standalone_title_fs,
        )
        plt.tight_layout()
        fig.savefig(os.path.join(dirs["figs"], "study_edf.pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    ecdf = np.arange(1, len(values) + 1) / len(values)

    # Interpolate values and ecdf for higher resolution
    interp_func = scipy.interpolate.interp1d(values, ecdf, kind="linear")
    high_res_values = np.linspace(values.min(), values.max(), num=1000)  # Increase resolution
    high_res_ecdf = interp_func(high_res_values)

    fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
    ax.step(high_res_values, high_res_ecdf, where="post", color="black")
    ax.set_xlabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel("Cumulative Proportion", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Empirical Distribution of Study Values", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_edf.pdf"), bbox_inches="tight")
    plt.close(fig)
