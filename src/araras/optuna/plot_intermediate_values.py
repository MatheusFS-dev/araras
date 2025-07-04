import os
from typing import Dict
import matplotlib.pyplot as plt
import optuna
import numpy as np

from .analyze import (
    PLOT_CFG,
)


def plot_intermediate_values(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Plot intermediate values reported during trials."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials for intermediate values plot.")
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
    fig.savefig(os.path.join(dirs["figs"], "study_intermediate_values.pdf"), bbox_inches="tight")
    plt.close(fig)
