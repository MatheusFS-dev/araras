"""
Last Edited: 14 July 2025
Description:
    This script demonstrates a detailed 
    header format with additional metadata.
"""
from araras.core import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    save_data_for_latex,
    draw_warning_box,
    save_plot,
)


def plot_spearman_correlation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    dirs: Dict[str, str],
    create_plotly: bool = False,
) -> None:
    """
    Generate and save Spearman correlation heatmap for numeric parameters and loss.

    This function computes rank-based correlations between all numeric parameters
    and the loss function, creating a heatmap visualization to identify
    relationships between parameters and their impact on optimization performance.

    Args:
        df (pd.DataFrame): Dataset containing numeric parameters and loss values
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs
        create_plotly (bool): Whether to save an interactive HTML version

    Returns:
        None: Saves correlation heatmap as pdf file
    """
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No numeric parameters for correlation plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_overall_correlation", "figs", create_plotly)
        plt.close(fig)
        return

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
    pfig_heatmap = None
    if create_plotly:
        import plotly.express as px

        pfig_heatmap = px.imshow(
            corr,
            x=display_cols,
            y=display_cols,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
        )

    save_plot(fig, dirs, "params_overall_correlation", "figs", create_plotly, pfig_heatmap)
    plt.close()

    # ——————————————————————————— Only loss correlation —————————————————————————— #
    # Extract correlations between each parameter and loss function only
    param_loss_corr = corr.loc[numeric_cols, "loss"]

    # Warn about NaN correlations (e.g., caused by constant parameters)
    nan_params = param_loss_corr[param_loss_corr.isna()].index.tolist()
    if nan_params:
        logger.warning(
            f"{YELLOW}NaN correlations detected for {nan_params}; skipping them.{RESET}"
        )

    # Drop NaN correlations so plotting works correctly
    param_loss_corr = param_loss_corr.dropna().sort_values(key=abs, ascending=False)

    # Save parameter-loss correlation data for LaTeX
    save_data_for_latex(
        {
            "parameter": param_loss_corr.index.tolist(),
            "correlation": param_loss_corr.values.tolist(),
        },
        "param_loss_correlations",
        dirs["data_correlations"],
    )

    if param_loss_corr.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No valid correlations to display.")
        plt.tight_layout()
        save_plot(fig, dirs, "params_study_value_correlations", "figs", create_plotly)
        plt.close(fig)
        return

    # Dynamic figure sizing based on parameter count and name length
    max_param_length = (
        max(len(str(param)) for param in param_loss_corr.index)
        if param_loss_corr.index.size > 0
        else 10
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

    # Calculate expanded x-axis limits to accommodate labels
    max_abs_corr = max(abs(param_loss_corr.min()), abs(param_loss_corr.max()))
    # Expand limits to ensure labels fit inside plot area
    x_limit = max_abs_corr * 2
    ax.set_xlim(-x_limit, x_limit)

    # Add labels with 3 decimal places for each bar
    for i, (bar, value) in enumerate(zip(bars, param_loss_corr.values)):
        # Position label closer to the end of the bar with reduced margin
        x_pos = value + (0.003 if value >= 0 else -0.003)
        ha = "left" if value >= 0 else "right"
        ax.text(x_pos, i, f"{value:.3f}", ha=ha, va="center", fontsize=PLOT_CFG.bar_value_fs)

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

    # Add grid for better readability
    ax.grid(True, axis="x", alpha=0.3, linestyle="-", linewidth=0.5)

    # Invert y-axis to show most correlated parameters at top
    ax.invert_yaxis()

    # Use subplots_adjust for better control over margins
    left_margin = max(0.2, max_param_length * 0.012)  # Dynamic left margin based on label length
    plt.subplots_adjust(left=left_margin, bottom=0.12, right=0.95, top=0.9)

    # Save parameter-loss correlation bar chart with better bounding box handling
    pfig_bar = None
    if create_plotly:
        import plotly.express as px

        pfig_bar = px.bar(
            x=param_loss_corr.values,
            y=param_loss_corr.index,
            orientation="h",
            range_x=[-x_limit, x_limit],
            color=param_loss_corr.values,
            color_continuous_scale="RdBu",
        )
        pfig_bar.update_layout(yaxis=dict(categoryorder="total ascending"))

    save_plot(fig, dirs, "params_study_value_correlations", "figs", create_plotly, pfig_bar)
    plt.close()
