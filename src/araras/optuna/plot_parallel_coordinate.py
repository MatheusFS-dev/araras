import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import numpy as np

from .analyze import (
    PLOT_CFG,
    get_param_display_name,
)


def plot_parallel_coordinate(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
) -> None:
    """Create a parallel coordinate plot for trials."""
    if not params:
        print("No parameters available for parallel coordinate plot.")
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={'value': 'loss'})
    if df.empty:
        print("No completed trials for parallel coordinate plot.")
        return

    cols = params + ["loss"]
    data = df[cols]

    # Normalize each column to 0-1 for plotting
    norm = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)

    color_vals = data["loss"].rank(method="dense", ascending=True)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(PLOT_CFG.numeric_subplot_size * len(cols), PLOT_CFG.box_subplot_height))
    for idx, (_, row) in enumerate(norm.iterrows()):
        ax.plot(range(len(cols)), row.values, color=cmap(color_vals.iloc[idx] / color_vals.max()), alpha=0.5)

    ax.set_xticks(range(len(cols)))
    labels = [get_param_display_name(c) if c != "loss" else PLOT_CFG.study_value_label for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=PLOT_CFG.x_tick_fs)
    ax.set_ylabel("Scaled Value", fontsize=PLOT_CFG.label_fs)
    ax.set_title("Parallel Coordinate Plot", pad=PLOT_CFG.title_pad)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_vals.min(), vmax=color_vals.max()))
    fig.colorbar(sm, ax=ax, label="Objective Rank")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_parallel_coordinate.pdf"), bbox_inches="tight")
    plt.close(fig)
