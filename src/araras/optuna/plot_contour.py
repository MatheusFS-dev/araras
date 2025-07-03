import os
from typing import List, Dict
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as vis

from .analyze import PLOT_CFG


def plot_contour(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_standalone: bool = False,
) -> None:
    """Generate contour plots for parameter pairs.

    This creates a single multipanel figure covering all provided parameters
    and optionally standalone figures for each pair of parameters.
    """
    if not params:
        print("No parameters available for contour plot.")
        return

    ax = vis.plot_contour(study, params=params)
    fig = ax.figure if hasattr(ax, "figure") else ax
    fig.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "params_contour.pdf"), bbox_inches="tight")
    plt.close(fig)

    if create_standalone:
        import itertools

        for p1, p2 in itertools.combinations(params, 2):
            ax = vis.plot_contour(study, params=[p1, p2])
            fig = ax.figure if hasattr(ax, "figure") else ax
            fig.tight_layout()
            filename = f"contour_{p1}_{p2}.pdf"
            fig.savefig(os.path.join(dirs["standalone_contours"], filename), bbox_inches="tight")
            plt.close(fig)
