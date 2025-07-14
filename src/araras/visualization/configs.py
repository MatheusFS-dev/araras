"""
Last Edited: 14 July 2025
Description:
    This script demonstrates a detailed 
    header format with additional metadata.
"""
from araras.core import *

import matplotlib.pyplot as plt
from cycler import cycler


def config_plt(style: str = "single-column") -> None:
    """
    Configure matplotlib rcParams for IEEE‑style figures

    Args:
        style (str): The figure style to use. Options are 'single-column' or
            'double-column'. Default is 'single-column'.

    Returns:
        None
    """
    if style == "single-column":
        figsize = (3.5, 2.5)
    elif style == "double-column":
        figsize = (7.2, 4.0)
    else:
        raise ValueError(f"Unknown style: {style!r}")

    plt.rcParams.update(
        {
            # Font settings
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.titleweight": "normal",
            "axes.titlepad": 6,
            "axes.labelpad": 4,
            # Line and marker settings
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "lines.markeredgewidth": 1.0,
            "axes.prop_cycle": cycler("color", ["k", "k", "k", "k"])
            * cycler("linestyle", ["-", "--", "-.", ":"]),
            # Tick settings
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.size": 3.5,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 2.0,
            "xtick.minor.width": 0.6,
            "ytick.minor.size": 2.0,
            "ytick.minor.width": 0.6,
            "axes.linewidth": 0.8,
            # Legend settings
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.borderaxespad": 0.5,
            # Grid and background
            "axes.grid": False,
            "grid.color": "gray",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            # Figure size and resolution
            "figure.figsize": figsize,
            "figure.dpi": 1200,
            "savefig.format": "pdf",
            "savefig.dpi": 1200,
            # Embed fonts in vector output
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
