"""
Last Edited: 14 July 2025
Description:
    This script demonstrates a detailed 
    header format with additional metadata.
"""
"""
Module plot_parallel_coordinate of analysis

Functions:
    - plot_parallel_coordinate: Create a parallel coordinate plot for trials.

Example:
    >>> from araras.optuna.analysis.plot_parallel_coordinate import plot_parallel_coordinate
    >>> plot_parallel_coordinate(...)
"""
from araras.core import *

import matplotlib.pyplot as plt
import optuna
import numpy as np

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    get_param_display_name,
    draw_warning_box,
    save_plot,
)


def plot_parallel_coordinate(
    study: optuna.Study,
    params: List[str],
    dirs: Dict[str, str],
    create_plotly: bool = False,
) -> None:
    """Create a parallel coordinate plot for trials.

    Parameters
    ----------
    create_plotly : bool
        Whether to save an interactive HTML version of the plot.
    """
    if not params:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No parameters available for parallel coordinate plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "study_parallel_coordinate", "figs", create_plotly)
        plt.close(fig)
        return

    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'").rename(columns={'value': 'loss'})
    if df.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for parallel coordinate plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "study_parallel_coordinate", "figs", create_plotly)
        plt.close(fig)
        return

    cols = params + ["loss"]
    data = df[cols]

    # Separate numeric and categorical columns
    numeric_cols = []
    categorical_cols = []

    for col in cols:
        if col == "loss" or data[col].dtype in ["int64", "float64"]:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Create normalized data
    norm = data.copy()

    # Keep track of numeric min/max and categorical mapping for annotations
    num_minmax: Dict[str, Tuple[float, float]] = {}
    cat_maps: Dict[str, Dict[Any, float]] = {}

    # Normalize numeric columns
    for col in numeric_cols:
        col_data = data[col]
        col_min = float(col_data.min())
        col_max = float(col_data.max())
        num_minmax[col] = (col_min, col_max)
        if col_max != col_min:
            norm[col] = (col_data - col_min) / (col_max - col_min)
        else:
            norm[col] = 0

    # Encode categorical columns
    for col in categorical_cols:
        unique_vals = list(data[col].unique())
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        cat_maps[col] = {val: idx / (len(unique_vals) - 1) if len(unique_vals) > 1 else 0 for val, idx in val_to_idx.items()}
        encoded = data[col].map(val_to_idx)
        if len(unique_vals) > 1:
            norm[col] = encoded / (len(unique_vals) - 1)
        else:
            norm[col] = 0

    color_vals = data["loss"].rank(method="dense", ascending=True)
    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots(
        figsize=(PLOT_CFG.numeric_subplot_size * len(cols), PLOT_CFG.box_subplot_height * 2)
    )
    try:
        for idx, (_, row) in enumerate(norm.iterrows()):
            ax.plot(
                range(len(cols)),
                row.values,
                color=cmap(color_vals.iloc[idx] / color_vals.max()),
                alpha=0.5,
                marker="o",
            )
    except Exception as e:
        draw_warning_box(ax, f"Could not generate plot: {e}")
        ax.set_title("Parallel Coordinate Plot", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
        plt.tight_layout()
        save_plot(fig, dirs, "study_parallel_coordinate", "figs", create_plotly)
        plt.close(fig)
        return

    ax.set_xticks(range(len(cols)))
    labels = [get_param_display_name(c) if c != "loss" else PLOT_CFG.study_value_label for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=PLOT_CFG.x_tick_fs)
    ax.set_ylabel("Scaled Value", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Parallel Coordinate Plot", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=color_vals.min(), vmax=color_vals.max()))
    fig.colorbar(sm, ax=ax, label="Objective Rank")
    ax.set_yticks(np.linspace(0, 1, 5))

    # Annotate axes with original values
    for i, col in enumerate(cols):
        if col in num_minmax:
            col_min, col_max = num_minmax[col]
            ax.text(i, 0, f"{col_min:.3g}", ha="center", va="top", fontsize=8)
            ax.text(i, 1, f"{col_max:.3g}", ha="center", va="bottom", fontsize=8)
        elif col in cat_maps:
            for val, pos in cat_maps[col].items():
                ax.text(i, pos, str(val), ha="center", va="center", fontsize=8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_parallel_coordinate(study, params=params)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "study_parallel_coordinate", "figs", create_plotly, pfig)
    plt.close(fig)
