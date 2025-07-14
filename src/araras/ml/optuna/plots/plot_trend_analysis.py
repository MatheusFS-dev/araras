from araras.core import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    format_title,
    get_param_display_name,
    save_data_for_latex,
    calculate_grid,
    draw_warning_box,
    save_plot,
)


def plot_trend_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
    dirs: Dict[str, str],
    param_name_mapping: Dict[str, str] = None,
    create_standalone: bool = False,
    create_plotly: bool = False,
) -> None:
    """
    Create a single comprehensive plot with trend analysis for parameter-loss relationships.

    This function generates a single plot with subplots showing the relationship between
    each numeric parameter and the loss function, fitting linear trends
    to identify parameter directions that improve performance.

    Args:
        df (pd.DataFrame): Dataset containing parameters and loss values
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs
        param_name_mapping (Dict[str, str]): Optional mapping for parameter display names
        create_standalone (bool): Whether to create standalone images for each parameter
        create_plotly (bool): Whether to save interactive HTML versions

    Returns:
        None: Saves single comprehensive trend plot as pdf file and trend statistics as CSV
    """
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No numeric parameters to analyze")
        plt.tight_layout()
        save_plot(fig, dirs, "params_trends", "figs", create_plotly)
        plt.close(fig)
        return

    stats = []

    # Calculate grid dimensions and adjust columns if needed
    max_cols = PLOT_CFG.max_cols
    n_plots = len(numeric_cols)
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

    # Analyze trend for each numeric parameter
    for plot_idx, col in enumerate(numeric_cols):
        row = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        ax = axes[row, col_idx]

        display_name = get_param_display_name(col, param_name_mapping)

        # Extract parameter values and corresponding loss values
        x = df[col].values
        y = df["loss"].values

        # Remove any infinite or NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        # Save scatter plot data for LaTeX
        save_data_for_latex(
            {"x": x_clean.tolist(), "y": y_clean.tolist()},
            f"trend_scatter_{col}",
            dirs["data_trends"],
        )

        # Check if we have enough valid data points
        if len(x_clean) < 2:
            logger.warning(
                f"{YELLOW}Not enough valid data points for parameter '{col}'. Skipping trend analysis.{RESET}"
            )
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\nfor {display_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PLOT_CFG.standalone_label_fs,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, display_name),
                fontsize=PLOT_CFG.title_fs,
                fontweight="bold",
                pad=PLOT_CFG.title_pad,
            )
            stats.append(
                {"Parameter": col, "Slope": np.nan, "Correlation": np.nan, "Status": "Insufficient data"}
            )
            continue

        # Check for constant values (no variance)
        if np.var(x_clean) == 0 or np.var(y_clean) == 0:
            logger.warning(
                f"{YELLOW}Parameter '{col}' or loss has no variance. Skipping trend analysis.{RESET}"
            )
            ax.text(
                0.5,
                0.5,
                f"No variance in\n{display_name} or loss",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PLOT_CFG.standalone_label_fs,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
            )
            ax.set_title(
                format_title(PLOT_CFG.param_title_tpl, display_name),
                fontsize=PLOT_CFG.title_fs,
                fontweight="bold",
                pad=PLOT_CFG.title_pad,
            )
            stats.append({"Parameter": col, "Slope": 0.0, "Correlation": 0.0, "Status": "No variance"})
            continue

        try:
            # Try to fit linear trend line using least squares
            slope, intercept = np.polyfit(x_clean, y_clean, 1)

            # Calculate correlation coefficient
            r = np.corrcoef(x_clean, y_clean)[0, 1]

            # Check if correlation is valid
            if np.isnan(r):
                r = 0.0

            fit_status = "Success"

        except (np.linalg.LinAlgError, ValueError) as e:
            logger_error.error(f"{RED}Error fitting trend line for parameter '{col}': {e}{RESET}")
            # Set default values
            slope = 0.0
            intercept = np.mean(y_clean) if len(y_clean) > 0 else 0.0
            r = 0.0
            fit_status = "Failed - using defaults"

        # Store statistics for this parameter
        stats.append(
            {
                "Parameter": col,
                "Slope": slope,
                "Correlation": r,
                "Status": fit_status,
                "Data_Points": len(x_clean),
                "X_Range": f"[{x_clean.min():.3f}, {x_clean.max():.3f}]" if len(x_clean) > 0 else "N/A",
                "Y_Range": f"[{y_clean.min():.3f}, {y_clean.max():.3f}]" if len(y_clean) > 0 else "N/A",
            }
        )

        # Create scatter plot first
        ax.scatter(x_clean, y_clean, s=10, edgecolor="black", linewidth=0.2, alpha=0.6)

        # Generate points for plotting fitted line CORRECTLY
        if len(x_clean) > 1 and np.var(x_clean) > 0 and abs(slope) > 1e-12:
            # Use the actual data range for x values
            x_min, x_max = x_clean.min(), x_clean.max()

            # Calculate corresponding y values using the fitted line equation: y = slope * x + intercept
            y_at_x_min = slope * x_min + intercept
            y_at_x_max = slope * x_max + intercept

            # Save trend line data for LaTeX
            save_data_for_latex(
                {"x": [x_min, x_max], "y": [y_at_x_min, y_at_x_max]},
                f"trend_line_{col}",
                dirs["data_trends"],
            )

            # Plot the line using only the endpoints to ensure correct visualization
            ax.plot([x_min, x_max], [y_at_x_min, y_at_x_max], linewidth=2, color="red", alpha=0.8)
        else:
            # For flat line case (slope ≈ 0)
            y_flat = intercept
            ax.axhline(y=y_flat, color="gray", linewidth=2, alpha=0.8)

        ax.set_xlabel(display_name, fontsize=PLOT_CFG.label_fs)
        ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.label_fs)
        ax.set_title(
            format_title(PLOT_CFG.param_title_tpl, display_name),
            fontsize=PLOT_CFG.title_fs,
            fontweight="bold",
            pad=PLOT_CFG.title_pad,
        )
        ax.grid(True, alpha=0.3)

        # Add text box with statistics
        stats_text = f"Slope: {slope:.6f}\n"  # Show more decimal places for slope
        stats_text += f"Correlation: {r:.4f}"
        if fit_status != "Success":
            stats_text += f"\nStatus: {fit_status}"

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

            # Recreate the scatter plot for standalone
            standalone_ax.scatter(x_clean, y_clean, s=15, edgecolor="black", linewidth=0.2, alpha=0.6)

            if len(x_clean) > 1 and np.var(x_clean) > 0 and abs(slope) > 1e-12:
                x_min, x_max = x_clean.min(), x_clean.max()
                y_at_x_min = slope * x_min + intercept
                y_at_x_max = slope * x_max + intercept
                standalone_ax.plot(
                    [x_min, x_max], [y_at_x_min, y_at_x_max], linewidth=2, color="red", alpha=0.8
                )
            else:
                y_flat = intercept
                standalone_ax.axhline(y=y_flat, color="gray", linewidth=2, alpha=0.8)

            standalone_ax.set_xlabel(display_name, fontsize=PLOT_CFG.standalone_label_fs)
            standalone_ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
            standalone_ax.set_title(
                format_title(
                    PLOT_CFG.trend_standalone_title_tpl,
                    display_name,
                ),
                fontsize=PLOT_CFG.standalone_title_fs,
                fontweight="bold",
                pad=PLOT_CFG.standalone_title_pad,
            )
            standalone_ax.grid(True, alpha=0.3)
            standalone_ax.tick_params(axis="x", labelsize=PLOT_CFG.standalone_x_tick_fs)
            standalone_ax.tick_params(axis="y", labelsize=PLOT_CFG.standalone_y_tick_fs)

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
                f"trend_{col}",
                "standalone_trends",
                create_plotly,
            )
            plt.close(standalone_fig)

    # Save trend statistics
    save_data_for_latex(
        pd.DataFrame(stats).to_dict("list"),
        "trend_statistics",
        dirs["data_trends"],
    )

    # Hide unused subplots if needed
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)

    # Adjust layout and save
    plt.suptitle(
        PLOT_CFG.trend_suptitle,
        fontsize=PLOT_CFG.suptitle_fs,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    pfig = None
    if create_plotly:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        pfig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[get_param_display_name(c, param_name_mapping) for c in numeric_cols],
        )
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1
            x = df[col].values
            y = df["loss"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            pfig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=col), row=row, col=col_idx)

    # Save the comprehensive trend plot
    save_plot(fig, dirs, "params_trends", "figs", create_plotly, pfig)
    plt.close()  # Close figure to free memory
