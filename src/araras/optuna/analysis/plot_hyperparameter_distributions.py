"""
Module plot_hyperparameter_distributions of analysis

Functions:
    - plot_hyperparameter_distributions: Generate and save distribution plots for numeric and categorical hyperparameters in separate figures.

Example:
    >>> from araras.optuna.analysis.plot_hyperparameter_distributions import plot_hyperparameter_distributions
    >>> plot_hyperparameter_distributions(...)
"""
from araras.commons import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
from scipy.stats import gaussian_kde

from .analyzer import (
    PLOT_CFG,
    format_title,
    get_param_display_name,
    format_numeric_value,
    save_data_for_latex,
    calculate_grid,
    draw_warning_box,
    save_plotly_html,
)

def plot_hyperparameter_distributions(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    dirs: Dict[str, str],
    param_name_mapping: Dict[str, str] = None,
    create_standalone: bool = False,
    save_plotly: bool = False,
) -> None:
    """
    Generate and save distribution plots for numeric and categorical hyperparameters in separate figures.

    For numeric parameters, a KDE curve is estimated prior to plotting. If the
    KDE computation fails (e.g., due to singular covariance or insufficient
    unique values), the parameter plot is replaced with a placeholder message so
    that the remaining plots can still be generated.

    Args:
        df (pd.DataFrame): DataFrame containing hyperparameter data
        numeric_cols (List[str]): List of numeric column names
        categorical_cols (List[str]): List of categorical column names
        dirs (Dict[str, str]): Dictionary of directory paths for saving plots
        param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
        create_standalone (bool): Whether to create standalone images for each parameter
    """

    # ———————————————————————— Numeric Parameters Figure ——————————————————————— #
    if numeric_cols:
        print(f"    Creating numeric distribution plots ({len(numeric_cols)} parameters)...")

        # Calculate grid dimensions and adjust columns if needed
        max_cols = PLOT_CFG.max_cols
        n_plots = len(numeric_cols)
        n_rows, n_cols = calculate_grid(
            n_plots,
            PLOT_CFG.numeric_subplot_size,
            PLOT_CFG.numeric_subplot_size,
            max_cols,
        )

        # Create grid layout for numeric parameters
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                PLOT_CFG.numeric_subplot_size * n_cols,
                PLOT_CFG.numeric_subplot_size * n_rows,
            ),
        )

        # Handle different array shapes
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each numeric parameter
        for plot_idx, col in enumerate(numeric_cols):
            row = plot_idx // n_cols
            col_idx = plot_idx % n_cols
            ax = axes[row, col_idx]

            display_name = get_param_display_name(col, param_name_mapping)
            values = df[col].dropna()

            # Skip parameters with no valid data
            if values.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No valid data\nfor {display_name}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=PLOT_CFG.label_fs,
                    style="italic",
                )
                ax.set_title(
                    format_title(PLOT_CFG.param_title_tpl, display_name),
                    fontsize=PLOT_CFG.title_fs,
                    fontweight="bold",
                    pad=PLOT_CFG.title_pad,
                )
                continue

            # Attempt KDE estimation first so that we can skip the plot entirely
            try:
                if values.nunique() < 2:
                    raise ValueError("insufficient unique values")

                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                kde_values = kde(x_range)
            except Exception as e:
                logger_error(f"{RED}Error generating KDE for {col}: {e}{RESET}")
                ax.text(
                    0.5,
                    0.5,
                    "No Data or Error Generating",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=PLOT_CFG.label_fs,
                    style="italic",
                )
                ax.set_title(
                    format_title(PLOT_CFG.param_title_tpl, display_name),
                    fontsize=PLOT_CFG.title_fs,
                    fontweight="bold",
                    pad=PLOT_CFG.title_pad,
                )
                continue

            # Main histogram
            n, bins, patches = ax.hist(
                values, bins=50, alpha=0.7, color="skyblue", edgecolor="navy", linewidth=0.8, density=True
            )

            # Save data for LaTeX
            bin_centers = (bins[:-1] + bins[1:]) / 2
            save_data_for_latex(
                {"x": bin_centers, "y": n},
                f"numeric_distribution_{col}",
                dirs["data_distributions"],
            )

            # KDE curve
            ax.plot(x_range, kde_values, color="darkblue", linewidth=2, alpha=0.8, label="KDE")

            # Save KDE data for LaTeX
            save_data_for_latex(
                {"x": x_range, "y": kde_values},
                f"numeric_kde_{col}",
                dirs["data_distributions"],
            )

            # Statistics
            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()

            # Format values using format_numeric_value
            mean_formatted = format_numeric_value(mean_val)
            median_formatted = format_numeric_value(median_val)
            std_formatted = format_numeric_value(std_val)

            # Add vertical lines with formatted labels
            ax.axvline(
                mean_val, color="red", linestyle="--", linewidth=2, alpha=0.8, label=f"Mean: {mean_formatted}"
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="-",
                linewidth=2,
                alpha=0.8,
                label=f"Median: {median_formatted}",
            )

            # Formatting
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, display_name),
                fontsize=PLOT_CFG.title_fs,
                fontweight="bold",
                pad=PLOT_CFG.title_pad,
            )
            ax.set_xlabel(display_name, fontsize=PLOT_CFG.label_fs)
            ax.set_ylabel(PLOT_CFG.density_label, fontsize=PLOT_CFG.label_fs)
            ax.legend(loc="upper right", fontsize=PLOT_CFG.legend_fs)
            ax.grid(True, alpha=0.3)

            # Statistics text box
            stats_text = f"Mean: {mean_formatted}\n"
            stats_text += f"Std: {std_formatted}\n"
            stats_text += f"Median: {median_formatted}"

            ax.text(
                0.05,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontsize=PLOT_CFG.annotation_fs,
                fontfamily="monospace",
            )

            # Create standalone image if requested
            if create_standalone:
                standalone_fig, standalone_ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

                # Recreate the plot for standalone
                standalone_ax.hist(
                    values, bins=50, alpha=0.7, color="skyblue", edgecolor="navy", linewidth=0.8, density=True
                )
                standalone_ax.plot(x_range, kde_values, color="darkblue", linewidth=2, alpha=0.8, label="KDE")
                standalone_ax.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Mean: {mean_formatted}",
                )
                standalone_ax.axvline(
                    median_val,
                    color="green",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Median: {median_formatted}",
                )

                standalone_ax.set_title(
                    format_title(
                        PLOT_CFG.dist_standalone_title_tpl,
                        display_name,
                    ),
                    fontsize=PLOT_CFG.standalone_title_fs,
                    fontweight="bold",
                    pad=PLOT_CFG.standalone_title_pad,
                )
                standalone_ax.set_xlabel(display_name, fontsize=PLOT_CFG.standalone_label_fs)
                standalone_ax.set_ylabel(PLOT_CFG.density_label, fontsize=PLOT_CFG.standalone_label_fs)
                standalone_ax.legend(loc="upper right", fontsize=PLOT_CFG.standalone_legend_fs)
                standalone_ax.grid(True, alpha=0.3)
                standalone_ax.tick_params(
                    axis="x",
                    labelsize=PLOT_CFG.standalone_x_tick_fs,
                )
                standalone_ax.tick_params(
                    axis="y",
                    labelsize=PLOT_CFG.standalone_y_tick_fs,
                )

                standalone_ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=standalone_ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                    fontsize=PLOT_CFG.standalone_legend_fs,
                    fontfamily="monospace",
                )

                plt.tight_layout()
                standalone_fig.savefig(
                    os.path.join(dirs["standalone_distributions"], f"numeric_distribution_{col}.pdf"),
                    bbox_inches="tight",
                )
                if save_plotly and dirs.get("plotly_standalone_distributions"):
                    save_plotly_html(
                        standalone_fig,
                        os.path.join(dirs["plotly_standalone_distributions"], f"numeric_distribution_{col}.html"),
                    )
                plt.close(standalone_fig)

        # Hide unused subplots if needed
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)

        # Adjust layout and save
        plt.suptitle(
            "Numeric Parameters Distributions",
            fontsize=PLOT_CFG.suptitle_fs,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        plt.savefig(
            os.path.join(dirs["figs"], "params_numeric_distributions.pdf"),
            bbox_inches="tight",
        )
        if save_plotly and dirs.get("plotly"):
            save_plotly_html(fig, os.path.join(dirs["plotly"], "params_numeric_distributions.html"))
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No numeric parameters found for distribution plotting.")
        ax.set_title(
            "Numeric Parameters Distributions",
            fontsize=PLOT_CFG.standalone_title_fs,
            pad=PLOT_CFG.title_pad,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(dirs["figs"], "params_numeric_distributions.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    # ——————————————————————— Categorical Parameters Figure ————————————————————— #
    if categorical_cols:
        print(f"    Creating categorical distribution plots ({len(categorical_cols)} parameters)...")

        # Calculate grid dimensions and adjust columns if needed
        max_cols = PLOT_CFG.max_cols
        n_plots = len(categorical_cols)
        n_rows, n_cols = calculate_grid(
            n_plots,
            PLOT_CFG.numeric_subplot_size,
            PLOT_CFG.numeric_subplot_size,
            max_cols,
        )

        # Create grid layout for categorical parameters
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                PLOT_CFG.numeric_subplot_size * n_cols,
                PLOT_CFG.numeric_subplot_size * n_rows,
            ),
        )

        # Handle different array shapes
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each categorical parameter
        for plot_idx, col in enumerate(categorical_cols):
            row = plot_idx // n_cols
            col_idx = plot_idx % n_cols
            ax = axes[row, col_idx]

            display_name = get_param_display_name(col, param_name_mapping)

            # Calculate category frequencies
            counts = df[col].value_counts()
            percentages = counts / counts.sum() * 100

            # Save data for LaTeX
            save_data_for_latex(
                {
                    "category": counts.index.tolist(),
                    "count": counts.values.tolist(),
                    "percentage": percentages.values.tolist(),
                },
                f"categorical_distribution_{col}",
                dirs["data_distributions"],
            )

            # Create enhanced bar chart
            bars = ax.bar(
                range(len(counts)),
                counts.values,
                color=plt.cm.Set3(np.linspace(0, 1, len(counts))),
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            # Customize x-axis labels
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(
                counts.index.astype(str),
                rotation=45,
                ha="right",
                fontsize=PLOT_CFG.x_tick_fs,
            )

            # Add value and percentage labels on bars
            max_count = max(counts.values)
            label_offset = max_count * 0.05

            for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages.values)):
                height = bar.get_height()

                # Format the count value using format_numeric_value
                count_formatted = format_numeric_value(count)
                pct_formatted = format_numeric_value(pct)

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + label_offset,
                    f"{count_formatted}\n({pct_formatted}%)",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=PLOT_CFG.bar_value_fs,
                )

            # Adjust y-axis to accommodate labels
            ax.set_ylim(0, max_count * 1.15)

            # Formatting
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, display_name),
                fontsize=PLOT_CFG.title_fs,
                fontweight="bold",
                pad=PLOT_CFG.title_pad,
            )
            ax.set_xlabel(display_name, fontsize=PLOT_CFG.label_fs)
            ax.set_ylabel(PLOT_CFG.count_label, fontsize=PLOT_CFG.label_fs)
            ax.grid(True, alpha=0.3, axis="y")

            # Create standalone image if requested
            if create_standalone:
                standalone_fig, standalone_ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

                # Recreate the plot for standalone
                standalone_bars = standalone_ax.bar(
                    range(len(counts)),
                    counts.values,
                    color=plt.cm.Set3(np.linspace(0, 1, len(counts))),
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1,
                )

                standalone_ax.set_xticks(range(len(counts)))
                standalone_ax.set_xticklabels(
                    counts.index.astype(str),
                    rotation=45,
                    ha="right",
                    fontsize=PLOT_CFG.x_tick_fs,
                )

                for i, (bar, count, pct) in enumerate(
                    zip(standalone_bars, counts.values, percentages.values)
                ):
                    height = bar.get_height()
                    standalone_ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + label_offset,
                        f"{count_formatted}\n({pct_formatted}%)",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=PLOT_CFG.standalone_legend_fs,
                    )

                standalone_ax.set_ylim(0, max_count * 1.15)
                standalone_ax.set_title(
                    format_title(
                        PLOT_CFG.dist_standalone_title_tpl,
                        display_name,
                    ),
                    fontsize=PLOT_CFG.standalone_title_fs,
                    fontweight="bold",
                    pad=PLOT_CFG.standalone_title_pad,
                )
                standalone_ax.set_xlabel(display_name, fontsize=PLOT_CFG.standalone_label_fs)
                standalone_ax.set_ylabel(PLOT_CFG.count_label, fontsize=PLOT_CFG.standalone_label_fs)
                standalone_ax.grid(True, alpha=0.3, axis="y")
                standalone_ax.tick_params(axis="x", labelsize=PLOT_CFG.standalone_x_tick_fs)
                standalone_ax.tick_params(axis="y", labelsize=PLOT_CFG.standalone_y_tick_fs)

                plt.tight_layout()
                standalone_fig.savefig(
                    os.path.join(dirs["standalone_distributions"], f"categorical_distribution_{col}.pdf"),
                    bbox_inches="tight",
                )
                if save_plotly and dirs.get("plotly_standalone_distributions"):
                    save_plotly_html(
                        standalone_fig,
                        os.path.join(dirs["plotly_standalone_distributions"], f"categorical_distribution_{col}.html"),
                    )
                plt.close(standalone_fig)

        # Hide unused subplots if needed
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)

        # Adjust layout and save
        plt.suptitle(
            "Categorical Parameters Distributions",
            fontsize=PLOT_CFG.suptitle_fs,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        plt.savefig(
            os.path.join(dirs["figs"], "params_categorical_distributions.pdf"),
            bbox_inches="tight",
        )
        if save_plotly and dirs.get("plotly"):
            save_plotly_html(fig, os.path.join(dirs["plotly"], "params_categorical_distributions.html"))
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No categorical parameters found for distribution plotting.")
        ax.set_title(
            "Categorical Parameters Distributions",
            fontsize=PLOT_CFG.standalone_title_fs,
            pad=PLOT_CFG.title_pad,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(dirs["figs"], "params_categorical_distributions.pdf"),
            bbox_inches="tight",
        )
        if save_plotly and dirs.get("plotly"):
            save_plotly_html(fig, os.path.join(dirs["plotly"], "params_categorical_distributions.html"))
        plt.close(fig)

    if not numeric_cols and not categorical_cols:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No parameters found for distribution plotting.")
        plt.tight_layout()
        fig.savefig(os.path.join(dirs["figs"], "params_no_distributions.pdf"), bbox_inches="tight")
        if save_plotly and dirs.get("plotly"):
            save_plotly_html(fig, os.path.join(dirs["plotly"], "params_no_distributions.html"))
        plt.close(fig)
