import os
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .analyze import PLOT_CFG, save_data_for_latex

def plot_spearman_correlation(df: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str]) -> None:
    """
    Generate and save Spearman correlation heatmap for numeric parameters and loss.

    This function computes rank-based correlations between all numeric parameters
    and the loss function, creating a heatmap visualization to identify
    relationships between parameters and their impact on optimization performance.

    Args:
        df (pd.DataFrame): Dataset containing numeric parameters and loss values
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs

    Returns:
        None: Saves correlation heatmap as pdf file
    """
    # Include loss column with numeric parameters for correlation analysis
    cols = numeric_cols + ["loss"]

    # Calculate Spearman rank correlation matrix (robust to non-linear relationships)
    corr = df[cols].corr(method="spearman")

    # Save correlation matrix data for LaTeX
    save_data_for_latex(
        corr.reset_index(),
        "spearman_correlation_matrix",
        dirs["data_correlations"],
    )

    # ———————————————————————— Complete correlation matrix ——————————————————————— #
    fig, ax = plt.subplots(
        figsize=(len(cols) * PLOT_CFG.heatmap_cell + 1, len(cols) * PLOT_CFG.heatmap_cell + 1)
    )

    # Create heatmap with correlation values mapped to colors (-1 to +1 range)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")

    cols = [col if col != "loss" else "Study Value" for col in cols]

    # Set axis labels to parameter names
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(
        cols,
        rotation=45,
        ha="right",
        fontsize=PLOT_CFG.x_tick_fs,
    )  # Rotate x-labels for readability
    ax.set_yticklabels(cols, fontsize=PLOT_CFG.y_tick_fs)

    # Add correlation values as text on each cell
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    # Add colorbar to show correlation scale
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title(PLOT_CFG.spearman_heatmap_title, pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)  # Descriptive title
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    # Save with high resolution
    fig.savefig(os.path.join(dirs["figs"], "params_overall_correlation.pdf"))
    plt.close()  # Close figure to free memory

    # ——————————————————————————— Only loss correlation —————————————————————————— #
    # Extract correlations between each parameter and loss function only
    param_loss_corr = corr.loc[numeric_cols, "loss"].sort_values(key=abs, ascending=False)

    # Save parameter-loss correlation data for LaTeX
    save_data_for_latex(
        {
            "parameter": param_loss_corr.index.tolist(),
            "correlation": param_loss_corr.values.tolist(),
        },
        "param_loss_correlations",
        dirs["data_correlations"],
    )

    # Create figure for parameter-loss correlation bar chart
    fig, ax = plt.subplots(
        figsize=(
            max(PLOT_CFG.corr_bar_min_width, len(numeric_cols) * PLOT_CFG.corr_bar_scale),
            PLOT_CFG.box_subplot_height,
        )
    )

    # Create color map based on correlation values (red for negative, blue for positive)
    colors = ["red" if x < 0 else "blue" for x in param_loss_corr.values]

    # Create horizontal bar chart for better parameter name readability
    bars = ax.barh(
        range(len(param_loss_corr)),
        param_loss_corr.values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Set y-axis labels to parameter names
    ax.set_yticks(range(len(param_loss_corr)))
    ax.set_yticklabels(param_loss_corr.index, fontsize=PLOT_CFG.y_tick_fs)

    # Add correlation values as text on each bar
    for i, (param, corr_val) in enumerate(param_loss_corr.items()):
        # Position text inside bar for better visibility
        text_x = corr_val * 0.5 if abs(corr_val) > 0.1 else corr_val + 0.05 * (1 if corr_val >= 0 else -1)
        ax.text(
            text_x,
            i,
            f"{corr_val:.3f}",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=PLOT_CFG.bar_value_fs,
        )

    # Add vertical line at x=0 for reference
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)

    # Add vertical lines at ±0.3 to highlight strong correlations
    ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    # Set axis labels and title
    ax.set_xlabel(PLOT_CFG.param_corr_xlabel)
    ax.set_ylabel(PLOT_CFG.param_corr_ylabel)
    ax.set_title(
        PLOT_CFG.param_corr_title,
        pad=PLOT_CFG.title_pad,
    )

    # Set x-axis limits with padding for better visualization
    max_abs_corr = max(abs(param_loss_corr.min()), abs(param_loss_corr.max()))
    ax.set_xlim(-max_abs_corr * 1.2, max_abs_corr * 1.2)

    # Add grid for better readability
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)

    # Invert y-axis to show most correlated parameters at top
    ax.invert_yaxis()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save parameter-loss correlation bar chart with high resolution
    fig.savefig(os.path.join(dirs["figs"], "params_study_value_correlations.pdf"))
    plt.close()  # Close figure to free memory


