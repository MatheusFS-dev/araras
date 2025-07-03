import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis

from .analyze import PLOT_CFG


def plot_slice(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
) -> None:
    """Create slice plots for each parameter."""
    if not params:
        print("No parameters available for slice plot.")
        return

    ax = vis.plot_slice(study, params=params)
    fig = ax.figure if hasattr(ax, "figure") else ax
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "params_slice.pdf"), bbox_inches="tight")
    plt.close(fig)

    if create_standalone:
        for p in params:
            ax = vis.plot_slice(study, params=[p])
            fig = ax.figure if hasattr(ax, "figure") else ax
            fig.tight_layout()
            fig.savefig(os.path.join(dirs["standalone_slices"], f"slice_{p}.pdf"), bbox_inches="tight")
            plt.close(fig)
