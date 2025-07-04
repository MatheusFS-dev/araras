import os
from typing import Dict
import matplotlib.pyplot as plt
import optuna
import numpy as np

from .analyze import (
    PLOT_CFG,
)


def plot_edf(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Plot the empirical distribution function of objective values."""
    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'")
    values = df["value"].astype(float).sort_values()
    if values.empty:
        print("No completed trials for EDF plot.")
        return

    ecdf = np.arange(1, len(values) + 1) / len(values)

    fig, ax = plt.subplots(figsize=PLOT_CFG.importance_size)
    ax.step(values, ecdf, where="post")
    ax.set_xlabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.label_fs)
    ax.set_ylabel("Cumulative Proportion", fontsize=PLOT_CFG.label_fs)
    ax.set_title("Empirical Distribution of Study Values", pad=PLOT_CFG.title_pad)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_edf.pdf"), bbox_inches="tight")
    plt.close(fig)
