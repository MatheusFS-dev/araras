# ———————————————————————————————————————————————————————————————————————————— #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ———————————————————————————————————————————————————————————————————————————— #

"""
This module contains functions to configure matplotlib rcParams for IEEE-style
figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

def config_plt(style: str = 'single-column') -> None:
    """
    Configure matplotlib rcParams for IEEE‑style figures
    
    Args:
        style (str): The figure style to use. Options are 'single-column' or
            'double-column'. Default is 'single-column'.
    
    Returns:
        None
    """
    if style == 'single-column':
        figsize = (3.5, 2.5)
    elif style == 'double-column':
        figsize = (7.2, 4.0)
    else:
        raise ValueError(f"Unknown style: {style!r}")

    plt.rcParams.update(
        {
            # Font settings
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
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

# def sample_plot() -> None:
#     """Generate a sample sine and cosine plot using IEEE‑style settings."""
#     x = np.linspace(0, 2 * np.pi, 100)
#     fig, ax = plt.subplots()
#     ax.plot(x, np.sin(x), label='sin(x)', marker='o',
#             mfc='none', mec='k', mew=1.0)
#     ax.plot(x, np.cos(x), label='cos(x)', marker='s',
#             mfc='none', mec='k', mew=1.0)
#     ax.set_xlabel('X (radians)')
#     ax.set_ylabel('Amplitude')
#     ax.set_title('Sine and Cosine')
#     ax.legend(loc='best')
#     ax.minorticks_on()
#     plt.tight_layout()
#     plt.show()
