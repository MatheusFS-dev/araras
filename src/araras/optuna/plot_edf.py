import os
from typing import Dict
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis

from .analyze import PLOT_CFG


def plot_edf(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Plot the empirical distribution function of objective values."""
    ax = vis.plot_edf(study)
    fig = ax.figure if hasattr(ax, "figure") else ax
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "study_edf.pdf"), bbox_inches="tight")
    plt.close(fig)
