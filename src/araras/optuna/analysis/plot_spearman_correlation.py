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
    # Calculate dynamic figure size based on content
    max_label_length = max(len(str(col)) for col in cols)
    base_size = len(cols) * PLOT_CFG.heatmap_cell + 1
    # Add extra space for rotated labels and colorbar
    fig_width = base_size + max(2, max_label_length * 0.08)
    fig_height = base_size + 1.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with correlation values mapped to colors (-1 to +1 range)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")

    # Replace 'loss' with 'Study Value' for display
    display_cols = [col if col != "loss" else "Study Value" for col in cols]

    # Set axis labels to parameter names
    ax.set_xticks(range(len(display_cols)))
    ax.set_yticks(range(len(display_cols)))
    ax.set_xticklabels(
        display_cols,
        rotation=45,
        ha="right",
        fontsize=PLOT_CFG.x_tick_fs,
    )
    ax.set_yticklabels(display_cols, fontsize=PLOT_CFG.y_tick_fs)

    # Add correlation values as text on each cell
    for i in range(len(display_cols)):
        for j in range(len(display_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    # Add colorbar with proper spacing
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08, shrink=0.8)
    cbar.ax.tick_params(labelsize=PLOT_CFG.y_tick_fs)

    plt.title(
        PLOT_CFG.spearman_heatmap_title, pad=PLOT_CFG.title_pad + 10, fontsize=PLOT_CFG.standalone_title_fs
    )

    # Use subplots_adjust for precise control over margins
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.85, top=0.9)

    # Save with bbox_inches='tight' to capture all elements
    fig.savefig(
        os.path.join(dirs["figs"], "params_overall_correlation.pdf"), bbox_inches="tight", pad_inches=0.3
    )
    plt.close()

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

    # Dynamic figure sizing based on parameter count and name length
    max_param_length = (
        max(len(str(param)) for param in param_loss_corr.index) if param_loss_corr.index.size > 0 else 10
    )
    fig_width = max(PLOT_CFG.corr_bar_min_width, len(numeric_cols) * PLOT_CFG.corr_bar_scale + 2)
    fig_height = max(PLOT_CFG.box_subplot_height, len(numeric_cols) * 0.4 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

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
    ax.set_xlabel(PLOT_CFG.param_corr_xlabel, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel(PLOT_CFG.param_corr_ylabel, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title(
        PLOT_CFG.param_corr_title,
        pad=PLOT_CFG.title_pad + 5,
        fontsize=PLOT_CFG.standalone_title_fs,
    )

    # Set x-axis limits with padding for better visualization
    max_abs_corr = max(abs(param_loss_corr.min()), abs(param_loss_corr.max()))
    ax.set_xlim(-max_abs_corr * 1.2, max_abs_corr * 1.2)

    # Add grid for better readability
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)

    # Invert y-axis to show most correlated parameters at top
    ax.invert_yaxis()

    # Use subplots_adjust for better control over margins
    left_margin = max(0.2, max_param_length * 0.012)  # Dynamic left margin based on label length
    plt.subplots_adjust(left=left_margin, bottom=0.12, right=0.95, top=0.9)

    # Save parameter-loss correlation bar chart with better bounding box handling
    fig.savefig(
        os.path.join(dirs["figs"], "params_study_value_correlations.pdf"), bbox_inches="tight", pad_inches=0.2
    )
    plt.close()
