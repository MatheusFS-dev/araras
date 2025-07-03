import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis

from .analyze import PLOT_CFG


def plot_rank(study: optuna.Study, params: List[str], dirs: Dict[str, str]) -> None:
    """Plot parameter relations colored by rank."""
    if not params:
        print("No parameters available for rank plot.")
        return

    ax = vis.plot_rank(study, params=params)
    fig = ax.figure if hasattr(ax, "figure") else ax
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_rank.pdf"), bbox_inches="tight")
    plt.close(fig)
