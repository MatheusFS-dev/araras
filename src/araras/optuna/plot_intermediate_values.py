import os
from typing import Dict
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis

from .analyze import PLOT_CFG


def plot_intermediate_values(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Plot intermediate values reported during trials."""
    try:
        ax = vis.plot_intermediate_values(study)
    except Exception as e:
        print(f"Could not create intermediate values plot: {e}")
        return
    fig = ax.figure if hasattr(ax, "figure") else ax
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_intermediate_values.pdf"), bbox_inches="tight")
    plt.close(fig)
