import os
from typing import Dict
import matplotlib.pyplot as plt
import optuna
import pandas as pd

from .analyze import PLOT_CFG


def plot_optimization_history(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Plot optimization history of the study."""
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df = df.query("state == 'COMPLETE'")
    if df.empty:
        print("No completed trials for optimization history plot.")
        return

    df = df.rename(columns={"value": "loss"})
    df = df.sort_values("number")
    loss = df["loss"].replace([float("inf"), float("-inf")], pd.NA)

    best = loss.cummin()

    fig, ax = plt.subplots(figsize=PLOT_CFG.importance_size)
    ax.scatter(df["number"], loss, color="blue", edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.plot(df["number"], best, color="red", linewidth=2, label="Best so far")

    ax.set_xlabel("Trial", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Optimization History", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    ax.legend(fontsize=PLOT_CFG.legend_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_optimization_history.pdf"), bbox_inches="tight")
    plt.close(fig)
