import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna

from .analyze import PLOT_CFG


def plot_terminator_improvement(
    study: optuna.Study,
    dirs: Dict[str, str],
    plot_error: bool = False,
    min_n_trials: int = 20,
) -> None:
    """Visualize estimated room for improvement for a study.

    This implementation mimics :func:`optuna.visualization.plot_terminator_improvement`
    without relying on Optuna's visualization module.  It estimates how much
    better the objective could still become based on the difference between the
    best value at each trial and the final best value. Optionally, the rolling
    standard deviation of the objective is plotted as an error curve.
    """

    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df = df.query("state == 'COMPLETE'")
    if df.empty:
        print("No completed trials for terminator improvement plot.")
        return

    df = df.sort_values("number")
    values = df["value"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    best_so_far = np.minimum.accumulate(values)
    final_best = best_so_far[-1]
    improvement = np.abs(best_so_far - final_best)

    if plot_error:
        error_curve = pd.Series(values).expanding().std().fillna(0).to_numpy()

    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=PLOT_CFG.importance_size)
    ax.plot(x, improvement, label="Improvement potential", color="blue", linewidth=2)

    if plot_error:
        ax.plot(x, error_curve, label="Evaluation error", color="orange", linewidth=2)

    ax.axvspan(0, min_n_trials, color="gray", alpha=0.15)
    ax.set_xlabel("Number of Trials", fontsize=PLOT_CFG.label_fs)
    ax.set_ylabel("Improvement", fontsize=PLOT_CFG.label_fs)
    ax.set_title("Terminator Improvement", pad=PLOT_CFG.title_pad)
    ax.legend(fontsize=PLOT_CFG.legend_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(os.path.join(dirs["figs"], "terminator_improvement.pdf"), bbox_inches="tight")
    plt.close(fig)

