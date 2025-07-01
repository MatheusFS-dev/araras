"""
Utility functions for analyzing Optuna study results.

This module provides comprehensive analysis tools for Optuna hyperparameter optimization studies,
generating statistical summaries, visualizations, and comparative analyses between best and worst
performing trials.

Functions:
    - analyze_study: Main analysis function generating all summaries and visualizations

Example usage:
    analyze_study(study=my_study, fig_dir="figures", table_dir="tables", top_frac=0.2)
"""

import os
import optuna
import numpy as np
from typing import *
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from araras.plot.configs import config_plt
from optuna.importance import get_param_importances


# Configure matplotlib for IEEE-style single-column figures
config_plt("single-column")

plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])

# ———————————————————————————————————————————————————————————————————————————— #
#                               Utility Functions                              #
# ———————————————————————————————————————————————————————————————————————————— #

def create_directories(table_dir: str) -> Dict[str, str]:
    """
    Create organized subdirectories for storing analysis outputs.
    
    This function establishes a structured directory hierarchy to organize
    different types of analysis outputs (figures and tables) into logical
    categories for easy navigation and interpretation.
    
    Args:
        fig_dir (str): Base directory path for saving figure outputs
        table_dir (str): Base directory path for saving table/CSV outputs
    
    Returns:
        Dict[str, str]: Dictionary mapping directory purpose to full path,
                       with keys like 'fig_boxplots', 'table_best', etc.
    """
    # Define organized subdirectory structure for different output types
    dirs = {
        "figs": os.path.join(table_dir, "figures"),
        "table_best": os.path.join(table_dir, "best"),
        "table_worst": os.path.join(table_dir, "worst"),
        "table_overall": os.path.join(table_dir, "overall"),
    }

    # Create each directory, allowing existing directories to remain unchanged
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def prepare_dataframe(study: optuna.Study) -> pd.DataFrame:
    """
    Extract and clean completed trial data from Optuna study.
    
    This function processes the raw Optuna study data to create a clean
    DataFrame suitable for analysis by filtering completed trials,
    renaming columns for clarity, and handling invalid loss values.
    
    Args:
        study (optuna.Study): Optuna study object containing trial results
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with columns for loss and all hyperparameters,
                     containing only successfully completed trials with valid loss values
    """
    # Extract trial data including trial metadata and hyperparameter values
    df = (
        study.trials_dataframe(attrs=("number", "value", "state", "params"))
        .query("state == 'COMPLETE'")  # Filter to only successfully completed trials
        .drop(columns=["number", "state"], errors="ignore")  # Remove unnecessary metadata columns
    )

    # Return empty DataFrame if no completed trials exist
    if df.empty:
        return df

    # Rename 'value' column to 'loss' for clarity in analysis
    df = df.rename(columns={"value": "loss"})

    # Handle infinite and NaN loss values by replacing with worst observed finite loss
    # This prevents infinite values from breaking statistical calculations
    finite = df["loss"].replace([np.inf, -np.inf], np.nan)  # Convert inf to NaN for processing
    worst = finite.max()  # Find the worst (highest) finite loss value
    df["loss"] = df["loss"].replace([np.inf, -np.inf], worst).fillna(worst)  # Replace inf/NaN with worst finite loss

    return df


def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Split DataFrame columns into numeric and categorical parameter types.
    
    This classification is essential for applying appropriate statistical
    methods and visualizations to different parameter types.
    
    Args:
        df (pd.DataFrame): DataFrame containing hyperparameters and loss values
    
    Returns:
        Tuple[List[str], List[str]]: Two lists containing (numeric_columns, categorical_columns)
                                   excluding the 'loss' column from numeric classification
    """
    # Identify numeric columns (excluding the loss column which is the target variable)
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "loss"]
    
    # Identify categorical/object columns (typically string-valued hyperparameters)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    
    return numeric_cols, categorical_cols


def get_trial_subsets(df: pd.DataFrame, top_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract best and worst performing trial subsets based on loss values.
    
    This function creates subsets for comparative analysis between
    high-performing and low-performing trials to identify parameter
    patterns that lead to better optimization results.
    
    Args:
        df (pd.DataFrame): Complete DataFrame with loss values and parameters
        top_frac (float): Fraction of trials to include in best/worst subsets (0 < top_frac < 1)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (best_trials, worst_trials) DataFrames
                                         containing top and bottom performing trials respectively
    """
    # Calculate number of trials for each subset, ensuring at least 1 trial
    n_top = max(1, int(len(df) * top_frac))
    
    # Get trials with smallest loss values (best performance)
    best = df.nsmallest(n_top, "loss")
    
    # Get trials with largest loss values (worst performance)
    worst = df.nlargest(n_top, "loss")
    
    return best, worst


def format_numeric_value(x: float) -> Union[int, float, str]:
    """
    Format numeric values with appropriate precision for readability.
    
    This function applies dynamic formatting rules to make numeric
    output more readable while preserving important precision information.
    
    Args:
        x (float): Numeric value to format
    
    Returns:
        Union[int, float, str]: Formatted value as integer (if whole number),
                               scientific notation (if very small), or rounded float
    """
    # Return special values unchanged
    if pd.isna(x) or np.isinf(x):
        return x

    # Convert to integer if value is effectively a whole number
    if abs(x - round(x)) < 1e-12:
        return int(round(x))

    # Use scientific notation for very small values
    if abs(x) < 1e-1:
        return f"{x:.2e}"

    # Round to 2 decimal places for regular values
    return float(round(x, 2))


def save_summary_tables(
    df: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    dirs: Dict[str, str],
) -> None:
    """
    Generate and save statistical summary tables for different trial subsets.

    This function creates comprehensive statistical summaries for overall,
    best-performing, and worst-performing trials, saving both numeric
    descriptive statistics and categorical frequency tables.

    Args:
        df (pd.DataFrame): Complete dataset with all trials
        best (pd.DataFrame): Subset of best-performing trials
        worst (pd.DataFrame): Subset of worst-performing trials
        numeric_cols (List[str]): List of numeric parameter column names
        categorical_cols (List[str]): List of categorical parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs

    Returns:
        None: Saves CSV files to specified directories
    """
    # Define datasets to analyze with their labels and corresponding data
    datasets = [("overall", df), ("best", best), ("worst", worst)]

    # Process each dataset (overall, best, worst trials)
    for label, subset in datasets:
        # Determine target directory based on dataset label
        dir_key = f"table_{label}"
        target_dir = dirs[dir_key]

        # Generate and save descriptive statistics for numeric parameters
        describe_numeric(subset, numeric_cols).to_csv(
            os.path.join(target_dir, f"{label}_numeric_summary.csv"), index=False
        )

        # Generate and save frequency tables for categorical parameters
        create_frequency_table(subset, categorical_cols).to_csv(
            os.path.join(target_dir, f"{label}_categorical_frequencies.csv"), index=False
        )


# ———————————————————————————————————————————————————————————————————————————— #
#                             Statistical Functions                            #
# ———————————————————————————————————————————————————————————————————————————— #


def describe_numeric(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Generate descriptive statistics for numeric hyperparameters.
    
    This function computes comprehensive statistical summaries including
    central tendency, variability, and distribution shape measures
    for numeric hyperparameters.
    
    Args:
        data (pd.DataFrame): DataFrame containing numeric parameters
        cols (List[str]): List of numeric column names to analyze
    
    Returns:
        pd.DataFrame: Statistics table with columns for Parameter, Mean, Std, Median,
                     25th percentile, 75th percentile, Min, Max values
    """
    stats = []

    # Calculate statistics for each numeric parameter
    for col in cols:
        arr = data[col]  # Extract parameter values as Series

        # Compute raw statistical measures
        raw = {
            "Parameter": col,
            "Mean": arr.mean(), # Average value
            "Std": arr.std(), # Standard deviation (variability)
            "Median": arr.median(), # Middle value (robust central tendency)
            "Min (25% quantile)": arr.quantile(0.25), # First quartile
            "Max (75% quantile)": arr.quantile(0.75), # Third quartile
            "Min (5% quantile)": arr.quantile(0.05), # 5th percentile (lower tail)
            "Max (95% quantile)": arr.quantile(0.95), # 95th percentile (upper tail)
        }

        # Apply formatting to all numeric values for readability
        formatted = {"Parameter": col}
        for k, v in raw.items():
            if k != "Parameter":
                formatted[k] = format_numeric_value(v)
        stats.append(formatted)

    return pd.DataFrame(stats)


def create_frequency_table(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Generate frequency tables for categorical hyperparameters.
    
    This function computes normalized frequencies and absolute counts
    for categorical parameters to understand their distribution patterns
    and identify commonly selected values.
    
    Args:
        data (pd.DataFrame): DataFrame containing categorical parameters
        cols (List[str]): List of categorical column names to analyze
    
    Returns:
        pd.DataFrame: Frequency table with columns for Parameter, Category,
                     Fraction (normalized frequency), and Count (absolute frequency)
    """
    rows = []

    # Process each categorical parameter
    for col in cols:
        # Calculate normalized frequencies (proportions) for each category
        counts = data[col].value_counts(normalize=True)

        # Create row for each category value
        for cat, frac in counts.items():
            rows.append(
                {
                    "Parameter": col, # Parameter name
                    "Category": cat, # Category value
                    "Fraction": round(frac, 4), # Normalized frequency (0-1)
                    "Count": int(data[col].value_counts()[cat]), # Absolute count
                }
            )

    return pd.DataFrame(rows)


def plot_hyperparameter_distributions(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], dirs: Dict[str, str]
) -> None:
    """
    Generate and save distribution plots for numeric and categorical hyperparameters in separate figures.

    Args:
        df (pd.DataFrame): DataFrame containing hyperparameter data
        numeric_cols (List[str]): List of numeric column names
        categorical_cols (List[str]): List of categorical column names
        dirs (Dict[str, str]): Dictionary of directory paths for saving plots
    """

    # ———————————————————————— Numeric Parameters Figure ——————————————————————— #
    if numeric_cols:
        print(f"Creating numeric parameters distribution plot ({len(numeric_cols)} parameters)...")

        # Calculate grid dimensions with max 4 columns
        max_cols = 4
        n_plots = len(numeric_cols)
        n_cols = min(n_plots, max_cols)
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division

        # Create grid layout for numeric parameters
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        # Handle different array shapes
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each numeric parameter
        for plot_idx, col in enumerate(numeric_cols):
            row = plot_idx // max_cols
            col_idx = plot_idx % max_cols
            ax = axes[row, col_idx]

            values = df[col].dropna()

            # Main histogram
            n, bins, patches = ax.hist(
                values, bins=50, alpha=0.7, color="skyblue", edgecolor="navy", linewidth=0.8, density=True
            )

            # KDE curve
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            kde_values = kde(x_range)
            ax.plot(x_range, kde_values, color="darkblue", linewidth=2, alpha=0.8, label="KDE")

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
            ax.set_title(f"{col}", fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            # Statistics text box
            stats_text = f"Mean: {mean_formatted}\n"
            stats_text += f"Std: {std_formatted}\n"
            stats_text += f"Median: {median_formatted}"

            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontsize=8,
                fontfamily="monospace",
            )

        # Hide unused subplots if needed
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // max_cols
            col_idx = idx % max_cols
            axes[row, col_idx].set_visible(False)

        # Adjust layout and save
        plt.suptitle("Numeric Parameters Distributions", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        plt.savefig(
            os.path.join(dirs["figs"], "params_numeric_distributions.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        print("No numeric parameters found for distribution plotting.")

    # ——————————————————————— Categorical Parameters Figure ————————————————————— #
    if categorical_cols:
        print(f"Creating categorical parameters distribution plot ({len(categorical_cols)} parameters)...")

        # Calculate grid dimensions with max 4 columns
        max_cols = 4
        n_plots = len(categorical_cols)
        n_cols = min(n_plots, max_cols)
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division

        # Create grid layout for categorical parameters
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        # Handle different array shapes
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each categorical parameter
        for plot_idx, col in enumerate(categorical_cols):
            row = plot_idx // max_cols
            col_idx = plot_idx % max_cols
            ax = axes[row, col_idx]

            # Calculate category frequencies
            counts = df[col].value_counts()
            percentages = counts / counts.sum() * 100

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
            ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")

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
                    fontsize=8,
                )

            # Adjust y-axis to accommodate labels
            ax.set_ylim(0, max_count * 1.15)

            # Formatting
            ax.set_title(f"{col}", fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

        # Hide unused subplots if needed
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // max_cols
            col_idx = idx % max_cols
            axes[row, col_idx].set_visible(False)

        # Adjust layout and save
        plt.suptitle("Categorical Parameters Distributions", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        plt.savefig(
            os.path.join(dirs["figs"], "params_categorical_distributions.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        print("No categorical parameters found for distribution plotting.")

    # Print summary
    if not numeric_cols and not categorical_cols:
        print("No parameters found for distribution plotting.")


def plot_param_importances(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """
    Generate and save parameter importance analysis.
    
    This function computes parameter importances using Optuna's built-in
    importance calculation and creates both a CSV table and bar chart
    visualization to identify which parameters most influence the objective.
    
    Args:
        study (optuna.Study): Optuna study object containing optimization history
        dirs (Dict[str, str]): Directory paths for saving outputs
    
    Returns:
        None: Saves importance table as CSV and bar chart as pdf
    """
    # Calculate parameter importances using Optuna's algorithm
    importances = get_param_importances(study)
    
    # Convert to DataFrame and sort by importance (descending)
    df_imp = pd.DataFrame(list(importances.items()), columns=["Parameter", "Importance"]).sort_values(
        "Importance", ascending=False
    )

    # Create bar chart visualization
    plt.figure(figsize=(6, 4))
    # Plot bars with parameter names on x-axis and importance values on y-axis
    plt.bar(df_imp["Parameter"], df_imp["Importance"], edgecolor="black")
    # Rotate parameter names for better readability
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")  # Importance score on y-axis
    plt.title("Hyperparameter Importances")  # Descriptive title
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    # Save with high resolution
    plt.savefig(os.path.join(dirs["figs"], "params_importances.pdf"))
    plt.close()  # Close figure to free memory


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

    # ———————————————————————— Complete correlation matrix ——————————————————————— #
    fig, ax = plt.subplots(figsize=(len(cols) * 0.5 + 1, len(cols) * 0.5 + 1))

    # Create heatmap with correlation values mapped to colors (-1 to +1 range)
    im = ax.imshow(corr, vmin=-1, vmax=1)

    cols = [col if col != "loss" else "Study Value" for col in cols]

    # Set axis labels to parameter names
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")  # Rotate x-labels for readability
    ax.set_yticklabels(cols)

    # Add correlation values as text on each cell
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    # Add colorbar to show correlation scale
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Spearman Correlation")  # Descriptive title
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    # Save with high resolution
    fig.savefig(os.path.join(dirs["figs"], "params_overall_correlation.pdf"))
    plt.close()  # Close figure to free memory

    # ——————————————————————————— Only loss correlation —————————————————————————— #
    # Extract correlations between each parameter and loss function only
    param_loss_corr = corr.loc[numeric_cols, "loss"].sort_values(key=abs, ascending=False)

    # Create figure for parameter-loss correlation bar chart
    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 0.6), 4))

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
    ax.set_yticklabels(param_loss_corr.index)

    # Add correlation values as text on each bar
    for i, (param, corr_val) in enumerate(param_loss_corr.items()):
        # Position text inside bar for better visibility
        text_x = corr_val * 0.5 if abs(corr_val) > 0.1 else corr_val + 0.05 * (1 if corr_val >= 0 else -1)
        ax.text(text_x, i, f"{corr_val:.3f}", ha="center", va="center", fontweight="bold", fontsize=9)

    # Add vertical line at x=0 for reference
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)

    # Add vertical lines at ±0.3 to highlight strong correlations
    ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    # Set axis labels and title
    ax.set_xlabel("Spearman Correlation with Study Value")
    ax.set_ylabel("Parameters")
    ax.set_title(
        "Parameter-Study Value Correlations\n(Red: Negative correlation = Lower values improve performance)"
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


def plot_parameter_boxplots(
    df: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    numeric_cols: List[str],
    dirs: Dict[str, str],
) -> None:
    """
    Create separate comprehensive boxplot comparisons for numeric parameters across trial subsets.

    Args:
        df (pd.DataFrame): Complete dataset with all trials
        best (pd.DataFrame): Subset of best-performing trials
        worst (pd.DataFrame): Subset of worst-performing trials
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs

    Returns:
        None: Saves separate boxplot files for numeric parameters
    """

    # ———————————————————————— Numeric Parameters Boxplots ——————————————————————— #
    if numeric_cols:
        print(f"Creating numeric parameters boxplots ({len(numeric_cols)} parameters)...")

        # Calculate grid dimensions with max 4 columns
        max_cols = 4
        n_plots = len(numeric_cols)
        n_cols = min(n_plots, max_cols)
        n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division

        # Create grid layout for numeric parameters
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Handle different array shapes
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Create boxplot for each numeric parameter
        for plot_idx, col in enumerate(numeric_cols):
            row = plot_idx // max_cols
            col_idx = plot_idx % max_cols
            ax = axes[row, col_idx]

            # Prepare data for boxplot: overall, best, worst trial subsets
            data = [df[col], best[col], worst[col]]
            labels = ["All trials", "Best trials", "Worst trials"]

            # Create boxplot with filled boxes for better visibility
            box_plot = ax.boxplot(data, labels=labels, patch_artist=True)

            # Color the boxes for better distinction
            colors = ["lightgray", "lightgreen", "lightcoral"]
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Styling
            ax.set_title(f"{col}", fontsize=14, fontweight="bold")
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            # Rotate x-axis labels for better readability
            ax.tick_params(axis="x", rotation=45)

        # Hide unused subplots if needed
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // max_cols
            col_idx = idx % max_cols
            axes[row, col_idx].set_visible(False)

        # Adjust layout and save
        plt.suptitle("Numeric Parameters Boxplots Comparison", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

        # Save the numeric parameters boxplot
        save_path = os.path.join(dirs["figs"], "params_numeric_boxplots.pdf")
        plt.savefig(save_path,  bbox_inches="tight")
        plt.close(fig)
    else:
        print("No numeric parameters found for boxplot analysis.")


def plot_trend_analysis(df: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str]) -> None:
    """
    Create a single comprehensive plot with trend analysis for parameter-loss relationships.

    This function generates a single plot with subplots showing the relationship between
    each numeric parameter and the loss function, fitting linear trends
    to identify parameter directions that improve performance.

    Args:
        df (pd.DataFrame): Dataset containing parameters and loss values
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs

    Returns:
        None: Saves single comprehensive trend plot as pdf file and trend statistics as CSV
    """
    if not numeric_cols:
        print("No numeric parameters to analyze")
        return

    stats = []

    # Calculate grid dimensions with max 4 columns
    max_cols = 4
    n_plots = len(numeric_cols)
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division

    # Create grid layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Handle different array shapes
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Analyze trend for each numeric parameter
    for plot_idx, col in enumerate(numeric_cols):
        row = plot_idx // max_cols
        col_idx = plot_idx % max_cols
        ax = axes[row, col_idx]

        # Extract parameter values and corresponding loss values
        x = df[col].values
        y = df["loss"].values

        # Remove any infinite or NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        # Check if we have enough valid data points
        if len(x_clean) < 2:
            print(f"Warning: Not enough valid data points for parameter '{col}'. Skipping trend analysis.")
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\nfor {col}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )
            ax.set_title(f"{col}", fontsize=14, fontweight="bold")
            stats.append(
                {"Parameter": col, "Slope": np.nan, "Correlation": np.nan, "Status": "Insufficient data"}
            )
            continue

        # Check for constant values (no variance)
        if np.var(x_clean) == 0 or np.var(y_clean) == 0:
            print(f"Warning: Parameter '{col}' or loss has no variance. Skipping trend analysis.")
            ax.text(
                0.5,
                0.5,
                f"No variance in\n{col} or loss",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
            )
            ax.set_title(f"{col}", fontsize=14, fontweight="bold")
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
            print(f"Warning: Could not fit trend line for parameter '{col}': {e}")
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

            # Plot the line using only the endpoints to ensure correct visualization
            ax.plot([x_min, x_max], [y_at_x_min, y_at_x_max], linewidth=2, color="red", alpha=0.8)

            # Verify the slope calculation is correct by checking the line's visual slope
            # Visual slope = (y_max - y_min) / (x_max - x_min) should equal our calculated slope
            visual_slope = (y_at_x_max - y_at_x_min) / (x_max - x_min)

            # Debug print to verify consistency (remove in production)
            if abs(visual_slope - slope) > 1e-10:
                print(
                    f"Warning: Slope mismatch for {col}. Calculated: {slope:.6f}, Visual: {visual_slope:.6f}"
                )
        else:
            # For flat line case (slope ≈ 0)
            y_flat = intercept
            ax.axhline(y=y_flat, color="gray", linewidth=2, alpha=0.8)

        # Determine trend direction for legend
        if abs(slope) < 1e-6:
            trend_legend = "No clear trend"
        elif slope > 0:
            trend_legend = "Higher parameter → HIGHER Study Value"
        else:
            trend_legend = "Higher parameter → LOWER Study Value"

        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Study Value", fontsize=10)
        ax.set_title(f"{col}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add legend with trend information
        if abs(slope) > 1e-6:
            ax.plot([], [], linewidth=2, color="red", label=trend_legend)  # Dummy plot for legend
        else:
            ax.plot([], [], linewidth=2, color="gray", label=trend_legend)  # Dummy plot for legend
        ax.legend(loc="best", fontsize=8)

        # Add text box with statistics
        stats_text = f"Slope: {slope:.6f}\n"  # Show more decimal places for slope
        stats_text += f"Correlation: {r:.4f}"
        if fit_status != "Success":
            stats_text += f"\nStatus: {fit_status}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
            fontfamily="monospace",
        )

    # Hide unused subplots if needed
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // max_cols
        col_idx = idx % max_cols
        axes[row, col_idx].set_visible(False)

    # Adjust layout and save
    plt.suptitle("Parameter-Study Value Trend Analysis", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    # Save the comprehensive trend plot
    save_path = os.path.join(dirs["figs"], "params_trends.pdf")
    plt.savefig(save_path,  bbox_inches="tight")
    plt.close()  # Close figure to free memory


def plot_optimal_ranges_analysis(
    df: pd.DataFrame, best: pd.DataFrame, numeric_cols: List[str], dirs: Dict[str, str]
) -> None:
    """
    Create a single comprehensive visualization showing optimal parameter ranges based on best-performing trials.

    This function generates a single plot with subplots for each parameter showing the distribution of parameters
    in all trials versus best trials, with indicators for conservative and aggressive
    optimal ranges, plus the median of best trials.

    Args:
        df (pd.DataFrame): Complete dataset with all trials
        best (pd.DataFrame): Subset of best-performing trials
        numeric_cols (List[str]): List of numeric parameter column names
        dirs (Dict[str, str]): Directory paths for saving outputs

    Returns:
        None: Saves the optimal ranges visualization to fig_ranges directory
    """
    if not numeric_cols:
        print("No numeric parameters to analyze")
        return

    # Process all parameters, even those with insufficient data
    ranges_data = []

    for col in numeric_cols:
        best_values = best[col].dropna()  # Remove NaN values
        all_values = df[col].dropna()  # Remove NaN values

        # Always add the parameter, but mark status for plotting
        param_data = {
            "parameter": col,
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

        ranges_data.append(param_data)

    # Calculate grid dimensions with max 4 columns
    max_cols = 4
    n_plots = len(numeric_cols)  # Use all parameters, not just valid ones
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + max_cols - 1) // max_cols  # Ceiling division

    # Create grid layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Handle different array shapes
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    plottable_count = 0
    for plot_idx, data in enumerate(ranges_data):
        row = plot_idx // max_cols
        col_idx = plot_idx % max_cols
        ax = axes[row, col_idx]

        col = data["parameter"]

        if not data["plottable"]:
            # Create blank graph with error message
            ax.text(
                0.5,
                0.5,
                f"Parameter: {col}\n\n"
                f"Analysis not possible\n\n"
                f"Reason:\n{data['error_message']}\n\n"
                f"Data points:\n"
                f"All trials: {len(data['all_values'])}\n"
                f"Best trials: {len(data['best_values'])}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
                linespacing=1.5,
            )
            ax.set_title(f"{col} (No Analysis)", fontsize=14, fontweight="bold", color="red")
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Analysis not available", fontsize=10)
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
                ax.set_title(f"{col}", fontsize=14, fontweight="bold", color="green")
                ax.set_xlabel(col, fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)

                # Add text box with statistics - format values safely
                def safe_format(value):
                    return format_numeric_value(value) if np.isfinite(value) else "N/A"

                stats_text = f"25%-75% : [{safe_format(data['conservative_min'])}, {safe_format(data['conservative_max'])}]\n"
                stats_text += f"5%-95% : [{safe_format(data['aggressive_min'])}, {safe_format(data['aggressive_max'])}]"

                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=8,
                    fontfamily="monospace",
                )

            except Exception as e:
                print(f"Error plotting parameter '{col}': {e}")
                # Create an error plot but still show the parameter
                ax.text(
                    0.5,
                    0.5,
                    f"Parameter: {col}\n\n"
                    f"Plotting error occurred\n\n"
                    f"Error details:\n{str(e)[:100]}...",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
                    linespacing=1.5,
                )
                ax.set_title(f"{col} (Error)", fontsize=14, fontweight="bold", color="red")
                ax.set_xlabel(col, fontsize=10)
                ax.set_ylabel("Error occurred", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xticks([])
                ax.set_yticks([])

    # Hide unused subplots if needed
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // max_cols
        col_idx = idx % max_cols
        axes[row, col_idx].set_visible(False)

    # Adjust layout and save
    plt.suptitle(
        f"Parameter Optimal Ranges Analysis ({plottable_count}/{len(numeric_cols)} parameters analyzed)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    # Save the comprehensive plot
    save_path = os.path.join(dirs["figs"], "params_optimal_ranges.pdf")
    plt.savefig(save_path,  bbox_inches="tight")
    plt.close()


# ———————————————————————————————————————————————————————————————————————————— #
#                    Main Function for the Analysis Pipeline                   #
# ———————————————————————————————————————————————————————————————————————————— #

def analyze_study(
    study: optuna.Study,
    table_dir: str,
    top_frac: float = 0.2,
) -> None:
    """
    Comprehensive analysis of Optuna hyperparameter optimization study results.
    
    This main function orchestrates a complete analysis pipeline that generates
    statistical summaries, visualizations, and comparative analyses to understand
    hyperparameter optimization performance and identify important parameter patterns.
    
    The analysis includes:
    - Parameter importance rankings and visualizations
    - Statistical summaries for overall, best, and worst performing trials
    - Distribution comparisons and trend analysis
    - Correlation analysis between parameters and performance
    - Statistical significance tests comparing high vs low performing trials
    - Comprehensive visualizations for all parameter types
    
    Args:
        study (optuna.Study): Optuna Study object containing completed optimization trials
        table_dir (str): Base directory path for saving CSV tables.
        top_frac (float): Fraction of trials to include in best/worst subsets for comparison.
                         Should be between 0 and 1 (default: 0.2 for top/bottom 20%)

    Returns:
        None: Saves all analysis outputs to specified directories and prints progress messages
    """
    print("\n\nAnalyzing study...")

    # Create organized directory structure for different output types
    dirs = create_directories(table_dir)

    # Extract and clean completed trial data from the study
    df = prepare_dataframe(study)
    if df.empty:
        print("No completed trials to analyze.")
        return

    # Classify parameters by type and create performance-based subsets
    numeric_cols, categorical_cols = classify_columns(df)
    best, worst = get_trial_subsets(df, top_frac)

    # Generate comprehensive statistical summary tables and plots
    print("Generating summary tables...")
    save_summary_tables(df, best, worst, numeric_cols, categorical_cols, dirs)
    print("Creating hyperparameter distribution plots...")
    plot_hyperparameter_distributions(df, numeric_cols, categorical_cols, dirs)
    print("Calculating parameter importances...")
    plot_param_importances(study, dirs)
    print("Analyzing Spearman correlations...")
    plot_spearman_correlation(df, numeric_cols, dirs)
    print("Creating boxplots for parameter distributions...")
    plot_parameter_boxplots(df, best, worst, numeric_cols, dirs)
    print("Performing trend analysis...")
    plot_trend_analysis(df, numeric_cols, dirs)
    print("Creating optimal ranges analysis...")
    plot_optimal_ranges_analysis(df, best, numeric_cols, dirs)
    
    print("Analysis complete! All results saved to the specified directories.")
