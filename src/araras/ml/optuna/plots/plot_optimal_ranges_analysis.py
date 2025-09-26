from araras.core import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from araras.ml.optuna.analyzer import PLOT_CFG
from araras.ml.optuna.analysis_utils import (
    format_title,
    get_param_display_name,
    calculate_grid,
    draw_warning_box,
    save_plot,
)

from araras.utils.misc import format_scientific


def plot_optimal_ranges_analysis(
    df: pd.DataFrame,
    best: pd.DataFrame,
    numeric_cols: List[str],
    dirs: Dict[str, str],
    param_name_mapping: Dict[str, str] = None,
    create_standalone: bool = False,
    create_plotly: bool = False,
) -> None:
    """Show conservative and aggressive value ranges for numeric parameters.

    Each subplot compares the full trial distribution with the subset of
    top-performing trials and highlights inter-quantile ranges plus the median
    value for the best trials.

    Args:
        df: DataFrame containing all recorded trials.
        best: DataFrame with the best performing trials.
        numeric_cols: Parameters to include in the range analysis.
        dirs: Mapping of directory identifiers to filesystem locations for
            outputs.
        param_name_mapping: Optional mapping from parameter names to
            presentation labels.
        create_standalone: Whether to generate individual figures per
            parameter.
        create_plotly: Whether to export interactive versions of the plots.

    Returns:
        None: Output files are saved beneath ``dirs["figs"]`` and related
        directories.
    """
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No numeric parameters to analyze")
        plt.tight_layout()
        save_plot(fig, dirs, "params_ranges", "figs", create_plotly)
        plt.close(fig)
        return

    def save_ranges_data_for_latex(
        col: str,
        all_values: pd.Series,
        best_values: pd.Series,
        conservative_min: float,
        conservative_max: float,
        aggressive_min: float,
        aggressive_max: float,
        best_median: float,
        data_dir: str,
    ) -> None:
        """
        Save ranges data for LaTeX plotting in separate files.

        Args:
            col (str): Parameter column name
            all_values (pd.Series): All trials data
            best_values (pd.Series): Best trials data
            conservative_min, conservative_max: 25%-75% range
            aggressive_min, aggressive_max: 5%-95% range
            best_median: Median of best trials
            data_dir (str): Directory to save data files
        """
        if data_dir is None:
            return

        import os

        # Save histogram data separately for each subset
        all_clean = all_values.dropna()
        best_clean = best_values.dropna()

        # Save all trials data
        if len(all_clean) > 0:
            all_df = pd.DataFrame({"value": all_clean.tolist(), "subset": ["all"] * len(all_clean)})
            all_filepath = os.path.join(data_dir, f"ranges_{col}_all_trials.csv")
            all_df.to_csv(all_filepath, index=False)

        # Save best trials data
        if len(best_clean) > 0:
            best_df = pd.DataFrame({"value": best_clean.tolist(), "subset": ["best"] * len(best_clean)})
            best_filepath = os.path.join(data_dir, f"ranges_{col}_best_trials.csv")
            best_df.to_csv(best_filepath, index=False)

        # Save range statistics
        ranges_stats = pd.DataFrame(
            [
                {
                    "parameter": col,
                    "conservative_min": conservative_min,
                    "conservative_max": conservative_max,
                    "aggressive_min": aggressive_min,
                    "aggressive_max": aggressive_max,
                    "best_median": best_median,
                    "all_trials_count": len(all_clean),
                    "best_trials_count": len(best_clean),
                }
            ]
        )
        ranges_filepath = os.path.join(data_dir, f"ranges_{col}_statistics.csv")
        ranges_stats.to_csv(ranges_filepath, index=False)

    # Process all parameters, even those with insufficient data
    ranges_data = []

    for col in numeric_cols:
        best_values = best[col].dropna()  # Remove NaN values
        all_values = df[col].dropna()  # Remove NaN values

        display_name = get_param_display_name(col, param_name_mapping)

        # Always add the parameter, but mark status for plotting
        param_data = {
            "parameter": col,
            "display_name": display_name,
            "all_values": all_values,
            "best_values": best_values,
            "plottable": True,
            "error_message": None,
        }

        # Check if we have enough valid data points
        if len(best_values) < 2:
            param_data["plottable"] = False
            param_data["error_message"] = f"Insufficient data in best trials\n({len(best_values)} points)"
        elif len(all_values) < 2:
            param_data["plottable"] = False
            param_data["error_message"] = f"Insufficient data in all trials\n({len(all_values)} points)"
        elif best_values.nunique() <= 1:
            param_data["plottable"] = False
            param_data["error_message"] = "No variance in best trials\n(all values identical)"
        else:
            # Calculate ranges only if data is valid
            param_data.update(
                {
                    "conservative_min": best_values.quantile(0.25),
                    "conservative_max": best_values.quantile(0.75),
                    "aggressive_min": best_values.quantile(0.05),
                    "aggressive_max": best_values.quantile(0.95),
                    "best_median": best_values.median(),
                }
            )

            # Save ranges data for LaTeX (fixed version)
            save_ranges_data_for_latex(
                col=col,
                all_values=all_values,
                best_values=best_values,
                conservative_min=param_data["conservative_min"],
                conservative_max=param_data["conservative_max"],
                aggressive_min=param_data["aggressive_min"],
                aggressive_max=param_data["aggressive_max"],
                best_median=param_data["best_median"],
                data_dir=dirs["data_ranges"],
            )

        ranges_data.append(param_data)

    # Calculate grid dimensions and adjust columns if needed
    max_cols = PLOT_CFG.max_cols
    n_plots = len(numeric_cols)  # Use all parameters, not just valid ones
    n_rows, n_cols = calculate_grid(
        n_plots,
        PLOT_CFG.numeric_subplot_size,
        PLOT_CFG.box_subplot_height,
        max_cols,
    )

    # Create grid layout
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

    plottable_count = 0
    for plot_idx, data in enumerate(ranges_data):
        row = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        ax = axes[row, col_idx]

        col = data["parameter"]
        display_name = data["display_name"]

        if not data["plottable"]:
            # Create blank graph with error message
            draw_warning_box(
                ax,
                f"Parameter: {display_name}\n\n"
                f"Analysis not possible\n\n"
                f"Reason:\n{data['error_message']}\n\n"
                f"Data points:\nAll: {len(data['all_values'])} / Best: {len(data['best_values'])}",
            )
            ax.set_title(
                f"{display_name} (No Analysis)",
                fontsize=PLOT_CFG.title_fs,
                fontweight="bold",
                color="red",
                pad=PLOT_CFG.title_pad,
            )
            ax.set_xlabel(display_name, fontsize=PLOT_CFG.label_fs)
            ax.set_ylabel("Analysis not available", fontsize=PLOT_CFG.label_fs)
            ax.grid(True, alpha=0.3)

            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

        else:
            # Plot normal analysis
            plottable_count += 1
            try:
                # Plot histograms with error handling
                ax.hist(
                    data["all_values"],
                    bins=min(50, max(10, len(data["all_values"]) // 2)),  # Adaptive bin count with minimum
                    alpha=0.3,
                    color="gray",
                    label="All trials",
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.hist(
                    data["best_values"],
                    bins=min(30, max(10, len(data["best_values"]) // 2)),  # Adaptive bin count with minimum
                    alpha=0.7,
                    color="green",
                    label="Best trials",
                    density=True,
                    edgecolor="darkgreen",
                    linewidth=0.8,
                )

                # Add range indicators only if values are finite
                if np.isfinite(data["conservative_min"]) and np.isfinite(data["conservative_max"]):
                    ax.axvline(
                        data["conservative_min"],
                        color="red",
                        linestyle="--",
                        alpha=0.8,
                        linewidth=2,
                        label="25%-75%",
                    )
                    ax.axvline(data["conservative_max"], color="red", linestyle="--", alpha=0.8, linewidth=2)

                    # Add shaded region for conservative range
                    ax.axvspan(
                        data["conservative_min"],
                        data["conservative_max"],
                        alpha=0.1,
                        color="red",
                        label="_nolegend_",
                    )

                if np.isfinite(data["aggressive_min"]) and np.isfinite(data["aggressive_max"]):
                    ax.axvline(
                        data["aggressive_min"],
                        color="blue",
                        linestyle=":",
                        alpha=0.8,
                        linewidth=2,
                        label="5%-95%",
                    )
                    ax.axvline(data["aggressive_max"], color="blue", linestyle=":", alpha=0.8, linewidth=2)

                    # Add shaded region for aggressive range
                    ax.axvspan(
                        data["aggressive_min"],
                        data["aggressive_max"],
                        alpha=0.05,
                        color="blue",
                        label="_nolegend_",
                    )

                # Best median
                if np.isfinite(data["best_median"]):
                    ax.axvline(
                        data["best_median"],
                        color="black",
                        linestyle="-",
                        alpha=0.9,
                        linewidth=2,
                        label="Median (best)",
                    )

                # Formatting
                ax.set_title(
                    format_title(PLOT_CFG.param_title_tpl, display_name),
                    fontsize=PLOT_CFG.title_fs,
                    fontweight="bold",
                    color="green",
                    pad=PLOT_CFG.title_pad,
                )
                ax.set_xlabel(display_name, fontsize=PLOT_CFG.label_fs)
                ax.set_ylabel(PLOT_CFG.density_label, fontsize=PLOT_CFG.label_fs)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=PLOT_CFG.legend_fs)

                # Add text box with statistics - format values safely
                def safe_format(value):
                    return format_scientific(value) if np.isfinite(value) else "N/A"

                stats_text = f"25%-75% : [{safe_format(data['conservative_min'])}, {safe_format(data['conservative_max'])}]\n"
                stats_text += (
                    f"5%-95% : [{safe_format(data['aggressive_min'])}, {safe_format(data['aggressive_max'])}]"
                )

                ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=PLOT_CFG.annotation_fs,
                    fontfamily="monospace",
                )

                # Create standalone image if requested
                if create_standalone:
                    standalone_fig, standalone_ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

                    # Recreate the plot for standalone
                    standalone_ax.hist(
                        data["all_values"],
                        bins=min(50, max(10, len(data["all_values"]) // 2)),
                        alpha=0.3,
                        color="gray",
                        label="All trials",
                        density=True,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    standalone_ax.hist(
                        data["best_values"],
                        bins=min(30, max(10, len(data["best_values"]) // 2)),
                        alpha=0.7,
                        color="green",
                        label="Best trials",
                        density=True,
                        edgecolor="darkgreen",
                        linewidth=0.8,
                    )

                    # Add range indicators
                    if np.isfinite(data["conservative_min"]) and np.isfinite(data["conservative_max"]):
                        standalone_ax.axvline(
                            data["conservative_min"],
                            color="red",
                            linestyle="--",
                            alpha=0.8,
                            linewidth=2,
                            label="25%-75%",
                        )
                        standalone_ax.axvline(
                            data["conservative_max"], color="red", linestyle="--", alpha=0.8, linewidth=2
                        )
                        standalone_ax.axvspan(
                            data["conservative_min"], data["conservative_max"], alpha=0.1, color="red"
                        )

                    if np.isfinite(data["aggressive_min"]) and np.isfinite(data["aggressive_max"]):
                        standalone_ax.axvline(
                            data["aggressive_min"],
                            color="blue",
                            linestyle=":",
                            alpha=0.8,
                            linewidth=2,
                            label="5%-95%",
                        )
                        standalone_ax.axvline(
                            data["aggressive_max"], color="blue", linestyle=":", alpha=0.8, linewidth=2
                        )
                        standalone_ax.axvspan(
                            data["aggressive_min"], data["aggressive_max"], alpha=0.05, color="blue"
                        )

                    if np.isfinite(data["best_median"]):
                        standalone_ax.axvline(
                            data["best_median"],
                            color="black",
                            linestyle="-",
                            alpha=0.9,
                            linewidth=2,
                            label="Median (best)",
                        )

                    standalone_ax.set_title(
                        format_title(
                            PLOT_CFG.ranges_standalone_title_tpl,
                            display_name,
                        ),
                        fontsize=PLOT_CFG.standalone_title_fs,
                        fontweight="bold",
                        pad=PLOT_CFG.standalone_title_pad,
                    )
                    standalone_ax.set_xlabel(display_name, fontsize=PLOT_CFG.standalone_label_fs)
                    standalone_ax.set_ylabel(PLOT_CFG.density_label, fontsize=PLOT_CFG.standalone_label_fs)
                    standalone_ax.grid(True, alpha=0.3)
                    standalone_ax.tick_params(axis="x", labelsize=PLOT_CFG.standalone_x_tick_fs)
                    standalone_ax.tick_params(axis="y", labelsize=PLOT_CFG.standalone_y_tick_fs)
                    standalone_ax.legend(loc="upper right", fontsize=PLOT_CFG.standalone_legend_fs)

                    standalone_ax.text(
                        0.05,
                        0.98,
                        stats_text,
                        transform=standalone_ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        fontsize=PLOT_CFG.standalone_legend_fs,
                        fontfamily="monospace",
                    )

                    plt.tight_layout()
                    save_plot(
                        standalone_fig,
                        dirs,
                        f"ranges_{col}",
                        "standalone_ranges",
                        create_plotly,
                    )
                    plt.close(standalone_fig)

            except Exception as e:
                logger_error.error(f"{RED}Error plotting parameter '{col}': {e}{RESET}")
                # Create an error plot but still show the parameter
                ax.text(
                    0.5,
                    0.5,
                    f"Parameter: {display_name}\n\n"
                    f"Plotting error occurred\n\n"
                    f"Error details:\n{str(e)[:100]}...",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=PLOT_CFG.label_fs,
                    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
                    linespacing=1.5,
                )
                ax.set_title(
                    f"{display_name} (Error)",
                    fontsize=PLOT_CFG.title_fs,
                    fontweight="bold",
                    color="red",
                    pad=PLOT_CFG.title_pad,
                )
                ax.set_xlabel(display_name, fontsize=PLOT_CFG.label_fs)
                ax.set_ylabel("Error occurred", fontsize=PLOT_CFG.label_fs)
                ax.grid(True, alpha=0.3)
                ax.set_xticks([])
                ax.set_yticks([])

    # Hide unused subplots if needed
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)

    # Adjust layout and save
    plt.suptitle(
        f"Parameter Optimal Ranges Analysis ({plottable_count}/{len(numeric_cols)} parameters analyzed)",
        fontsize=PLOT_CFG.suptitle_fs,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    pfig = None
    if create_plotly:
        import plotly.express as px

        long_data = []
        for col in numeric_cols:
            long_data.append(pd.DataFrame({"value": df[col].dropna(), "param": col, "subset": "all"}))
            long_data.append(pd.DataFrame({"value": best[col].dropna(), "param": col, "subset": "best"}))
        if long_data:
            long_df = pd.concat(long_data)
            pfig = px.histogram(
                long_df,
                x="value",
                color="subset",
                facet_col="param",
                facet_col_wrap=n_cols,
                barmode="overlay",
            )

    # Save the comprehensive plot
    save_plot(fig, dirs, "params_optimal_ranges", "figs", create_plotly, pfig)
    plt.close()
