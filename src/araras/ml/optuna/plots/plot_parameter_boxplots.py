"""
Last Edited: 14 July 2025
Description:
    Box plots for hyperparameter values.
"""
from araras.core import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    format_title,
    get_param_display_name,
    calculate_grid,
    draw_warning_box,
    save_plot,
)


def plot_parameter_boxplots(
    df: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    numeric_cols: List[str],
    dirs: Dict[str, str],
    param_name_mapping: Dict[str, str] = None,
    create_standalone: bool = False,
    create_plotly: bool = False,
) -> None:
    """
    Create separate comprehensive boxplot comparisons for numeric parameters across trial subsets.

    Args:
        df (pd.DataFrame): Complete dataset with all trials
        best (pd.DataFrame): Subset of best-performing trials
        worst (pd.DataFrame): Subset of worst-performing trials
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs
        param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
        create_standalone (bool): Whether to create standalone images for each parameter
        create_plotly (bool): Whether to save interactive HTML versions

    Returns:
        None: Saves separate boxplot files for numeric parameters
    """

    def save_boxplot_data_for_latex(
        col: str, all_data: pd.Series, best_data: pd.Series, worst_data: pd.Series, data_dir: str
    ) -> None:
        """
        Save boxplot data for LaTeX plotting in separate files for each subset.

        Args:
            col (str): Parameter column name
            all_data (pd.Series): All trials data
            best_data (pd.Series): Best trials data
            worst_data (pd.Series): Worst trials data
            data_dir (str): Directory to save data files
        """
        if data_dir is None:
            return

        import os

        # Clean the data (remove NaN values)
        all_clean = all_data.dropna()
        best_clean = best_data.dropna()
        worst_clean = worst_data.dropna()

        # Save each subset separately since they have different lengths
        subsets = {"all": all_clean, "best": best_clean, "worst": worst_clean}

        for subset_name, data in subsets.items():
            if len(data) > 0:
                # Create DataFrame with single column
                df_subset = pd.DataFrame({"value": data.tolist(), "subset": [subset_name] * len(data)})

                # Save to CSV
                filepath = os.path.join(data_dir, f"boxplot_{col}_{subset_name}.csv")
                df_subset.to_csv(filepath, index=False)

        # Also save summary statistics for each subset
        summary_data = []
        for subset_name, data in subsets.items():
            if len(data) > 0:
                summary_data.append(
                    {
                        "subset": subset_name,
                        "count": len(data),
                        "mean": data.mean(),
                        "std": data.std(),
                        "min": data.min(),
                        "q25": data.quantile(0.25),
                        "median": data.median(),
                        "q75": data.quantile(0.75),
                        "max": data.max(),
                    }
                )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_filepath = os.path.join(data_dir, f"boxplot_{col}_summary.csv")
            summary_df.to_csv(summary_filepath, index=False)

    # Main function logic
    if numeric_cols:
        # print(f"    Creating numeric boxplots ({len(numeric_cols)} parameters)...")

        # Calculate grid dimensions and adjust columns if needed
        max_cols = PLOT_CFG.max_cols
        n_plots = len(numeric_cols)
        n_rows, n_cols = calculate_grid(
            n_plots,
            PLOT_CFG.numeric_subplot_size,
            PLOT_CFG.box_subplot_height,
            max_cols,
        )

        # Create grid layout for numeric parameters
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                PLOT_CFG.numeric_subplot_size * n_cols,
                PLOT_CFG.box_subplot_height * n_rows,
            ),
        )

        # Handle different array shapes
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Create boxplot for each numeric parameter
        for plot_idx, col in enumerate(numeric_cols):
            row = plot_idx // n_cols
            col_idx = plot_idx % n_cols
            ax = axes[row, col_idx]

            display_name = get_param_display_name(col, param_name_mapping)

            # Prepare data for boxplot: overall, best, worst trial subsets
            data = [df[col], best[col], worst[col]]
            labels = ["All trials", "Best trials", "Worst trials"]

            if all(d.dropna().empty for d in data):
                draw_warning_box(ax, f"No valid data for {display_name}")
                ax.set_title(
                    format_title(PLOT_CFG.param_title_tpl, display_name),
                    fontsize=PLOT_CFG.title_fs,
                    pad=PLOT_CFG.title_pad,
                )
                continue

            # Save boxplot data for LaTeX (fixed version)
            save_boxplot_data_for_latex(col, df[col], best[col], worst[col], dirs["data_boxplots"])

            # Create boxplot with filled boxes for better visibility
            box_plot = ax.boxplot(data, labels=labels, patch_artist=True)

            # Color the boxes for better distinction
            colors = ["lightgray", "lightgreen", "lightcoral"]
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Styling
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, display_name),
                fontsize=PLOT_CFG.title_fs,
                fontweight="bold",
                pad=PLOT_CFG.title_pad,
            )
            ax.set_ylabel(display_name, fontsize=PLOT_CFG.label_fs)
            ax.grid(True, alpha=0.3, axis="y")

            # Rotate x-axis labels for better readability
            ax.tick_params(axis="x", rotation=45, labelsize=PLOT_CFG.x_tick_fs)

            # Create standalone image if requested
            if create_standalone:
                standalone_fig, standalone_ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

                # Recreate the boxplot for standalone
                standalone_box_plot = standalone_ax.boxplot(data, labels=labels, patch_artist=True)

                for patch, color in zip(standalone_box_plot["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                standalone_ax.set_title(
                    format_title(
                        PLOT_CFG.box_standalone_title_tpl,
                        display_name,
                    ),
                    fontsize=PLOT_CFG.standalone_title_fs,
                    fontweight="bold",
                    pad=PLOT_CFG.standalone_title_pad,
                )
                standalone_ax.set_ylabel(display_name, fontsize=PLOT_CFG.standalone_label_fs)
                standalone_ax.grid(True, alpha=0.3, axis="y")
                standalone_ax.tick_params(axis="x", rotation=45, labelsize=PLOT_CFG.standalone_x_tick_fs)
                standalone_ax.tick_params(axis="y", labelsize=PLOT_CFG.standalone_y_tick_fs)

                plt.tight_layout()
                save_plot(
                    standalone_fig,
                    dirs,
                    f"boxplot_{col}",
                    "standalone_boxplots",
                    create_plotly,
                )
                plt.close(standalone_fig)

        # Hide unused subplots if needed
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)

        # Adjust layout and save
        plt.suptitle(
            "Numeric Parameters Boxplots Comparison",
            fontsize=PLOT_CFG.suptitle_fs,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        # Create Plotly version
        pfig = None
        if create_plotly:
            import plotly.express as px

            long_data = []
            for col in numeric_cols:
                long_data.append(pd.DataFrame({"value": df[col].dropna(), "subset": "all", "param": col}))
                long_data.append(pd.DataFrame({"value": best[col].dropna(), "subset": "best", "param": col}))
                long_data.append(
                    pd.DataFrame({"value": worst[col].dropna(), "subset": "worst", "param": col})
                )
            if long_data:
                long_df = pd.concat(long_data)
                pfig = px.box(
                    long_df,
                    x="subset",
                    y="value",
                    facet_col="param",
                    facet_col_wrap=n_cols,
                    points="outliers",
                )

        # Save the numeric parameters boxplot
        save_plot(fig, dirs, "params_numeric_boxplots", "figs", create_plotly, pfig)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No numeric parameters found for boxplot analysis.")
        ax.set_title(
            "Numeric Parameters Boxplots Comparison",
            fontsize=PLOT_CFG.standalone_title_fs,
            pad=PLOT_CFG.title_pad,
        )
        plt.tight_layout()
        save_plot(fig, dirs, "params_numeric_boxplots", "figs", create_plotly)
        plt.close(fig)
