import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis

from .analyze import PLOT_CFG


def plot_parallel_coordinate(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
) -> None:
    """Create a parallel coordinate plot for trials."""
    if not params:
        print("No parameters available for parallel coordinate plot.")
        return

    ax = vis.plot_parallel_coordinate(study, params=params)
    fig = ax.figure if hasattr(ax, "figure") else ax
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_parallel_coordinate.pdf"), bbox_inches="tight")
    plt.close(fig)
