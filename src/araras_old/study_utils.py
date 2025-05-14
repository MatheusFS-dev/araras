"""
Utility functions for analyzing and saving Optuna study results.

Functions:
    - save_trial_params_to_file: Saves trial parameters and metadata to a text file.
    - analyze_study: Analyzes an Optuna study, generating summary tables and visualizations.

Example usage:
    save_trial_params_to_file("trial1.txt", {"lr": 0.01}, trial_id="1", loss="0.15")
    analyze_study(study=my_study, fig_dir="figures", table_dir="tables", top_frac=0.2)
"""

import os
import math
import numpy as np
import pandas as pd
from typing import *
from IPython.display import display, HTML
import optuna
import matplotlib.pyplot as plt


def save_trial_params_to_file(filepath: str, params: dict[str, float], **kwargs: str) -> None:
    """
    Save Optuna trial parameters and associated metadata to a text file.

    Logic:
        -> Open file for writing
        -> Write key-value metadata (from kwargs)
        -> Write trial parameters section with indentation

    Args:
        filepath (str): Path where the parameter file should be saved.
        params (dict[str, float]): Dictionary of trial hyperparameters.
        **kwargs (str): Additional information such as trial ID, rank, or loss.

    Returns:
        None

    Example:
        save_trial_params_to_file("trial1.txt", {"lr": 0.01}, trial_id="1", loss="0.15")
    """
    with open(filepath, "w") as file:
        # Write metadata key-value pairs first
        file.writelines(f"{k}: {v}\n" for k, v in kwargs.items())

        # Write trial hyperparameters
        file.write("Parameters:\n")
        file.writelines(f"  {k}: {v}\n" for k, v in params.items())


def analyze_study(
    study: optuna.Study,
    fig_dir: str,
    table_dir: Optional[str] = None,
    top_frac: float = 0.2,
) -> None:
    """
    Analyze an Optuna study by computing summary tables and visualizing parameter distributions.

    Logic:
        -> Create output directories
        -> Count failed trials
        -> Filter and clean completed trials
        -> Categorize parameters (numeric or categorical)
        -> Create summary statistics for all, top-N, and bottom-N trials
        -> Save these tables to CSV files
        -> Display summaries in Jupyter/IPython
        -> Plot histograms and bar charts of parameters

    Args:
        study (optuna.Study): An Optuna Study containing optimization results.
        fig_dir (str): Directory to save visualizations.
        table_dir (Optional[str]): Directory to save CSV tables. Defaults to fig_dir.
        top_frac (float): Fraction of top trials to use for summaries (0 < top_frac < 1).

    Returns:
        None

    Example:
        analyze_study(study=my_study, fig_dir="figures", table_dir="tables", top_frac=0.2)
    """
    os.makedirs(fig_dir, exist_ok=True)  # Ensure figure output directory exists
    if table_dir is None:
        table_dir = fig_dir  # Use same directory if no table_dir provided
    os.makedirs(table_dir, exist_ok=True)  # Ensure table output directory exists

    # Step 1: Count failed trials (not in 'COMPLETE' state)
    failed = sum(1 for t in study.trials if t.state != t.state.COMPLETE)
    print(f"Number of failed trials: {failed}\n")

    # Step 2: Extract completed trial data into a DataFrame
    df = (
        study.trials_dataframe(attrs=("number", "value", "state", "params"))
        .query("state == 'COMPLETE'")  # Keep only completed trials
        .drop(columns=["number", "state"], errors="ignore")  # Remove unnecessary columns
    )
    if df.empty:
        print("No completed trials to analyze.")
        return

    # Rename 'value' to 'loss' for clarity
    df = df.rename(columns={"value": "loss"})

    # Replace inf/nan loss values with the worst observed finite loss
    finite = df["loss"].replace([np.inf, -np.inf], np.nan)
    worst = finite.max()
    df["loss"] = df["loss"].replace([np.inf, -np.inf], worst).fillna(worst)

    # Step 3: Split columns by type
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "loss"]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # Step 4: Helper functions for summaries
    def describe_numeric(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Return descriptive stats for each numeric column."""
        stats = []
        for col in cols:
            arr = data[col]
            stats.append(
                {
                    "Parameter": col,
                    "Mean": arr.mean(),
                    "Std": arr.std(),
                    "Min": arr.min(),
                    "25%": arr.quantile(0.25),
                    "Median": arr.median(),
                    "75%": arr.quantile(0.75),
                    "Max": arr.max(),
                }
            )
        return pd.DataFrame(stats)

    def freq_table(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Return normalized frequencies for categorical columns."""
        rows = []
        for col in cols:
            counts = data[col].value_counts(normalize=True)
            for cat, frac in counts.items():
                rows.append(
                    {
                        "Parameter": col,
                        "Category": cat,
                        "Fraction": frac,
                        "Count": int(data[col].value_counts()[cat]),
                    }
                )
        return pd.DataFrame(rows)

    # Select top and bottom performing trials
    n_top = max(1, int(len(df) * top_frac))
    best = df.nsmallest(n_top, "loss")  # Top trials with lowest loss
    worst_df = df.nlargest(n_top, "loss")  # Bottom trials with highest loss

    # Generate and save numeric/categorical summaries for overall, best, and worst trials
    for label, subset in [("overall", df), ("best", best), ("worst", worst_df)]:
        describe_numeric(subset, numeric_cols).to_csv(
            os.path.join(table_dir, f"{label}_numeric_summary.csv"), index=False
        )
        freq_table(subset, categorical_cols).to_csv(
            os.path.join(table_dir, f"{label}_categorical_frequencies.csv"), index=False
        )

    # Display tables in IPython
    display(HTML("<h3>Overall Numeric Hyperparameter Summary</h3>"))
    display(describe_numeric(df, numeric_cols))
    display(HTML("<h3>Overall Categorical Frequencies</h3>"))
    display(freq_table(df, categorical_cols))

    display(HTML(f"<h3>Top {int(top_frac*100)}% Numeric Summary (lowest loss)</h3>"))
    display(describe_numeric(best, numeric_cols))
    display(HTML(f"<h3>Top {int(top_frac*100)}% Categorical Frequencies</h3>"))
    display(freq_table(best, categorical_cols))

    display(HTML(f"<h3>Bottom {int(top_frac*100)}% Numeric Summary (highest loss)</h3>"))
    display(describe_numeric(worst_df, numeric_cols))
    display(HTML(f"<h3>Bottom {int(top_frac*100)}% Categorical Frequencies</h3>"))
    display(freq_table(worst_df, categorical_cols))

    # Step 5: Create visualizations
    max_rows = 4
    total_plots = len(numeric_cols) + len(categorical_cols)

    # Determine number of columns so that rows ≤ max_rows
    if total_plots > max_rows:
        cols = math.ceil(total_plots / max_rows)
    else:
        cols = total_plots or 1
    rows = math.ceil(total_plots / cols)

    # Compute figure size
    fig_width = cols * 4   # e.g. 4 inches per column
    fig_height = rows * 4  # e.g. 4 inches per row
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easy iteration
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Plot numeric parameter histograms
    for i, col in enumerate(numeric_cols):
        ax = axes_flat[i]
        ax.hist(df[col], bins=30, edgecolor="black", color="black")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    # Plot categorical parameter bar charts
    for j, col in enumerate(categorical_cols, start=len(numeric_cols)):
        ax = axes_flat[j]
        counts = df[col].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="black", edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")

    # Remove unused subplots if total_plots doesn't fill the grid
    for k in range(total_plots, len(axes_flat)):
        fig.delaxes(axes_flat[k])

    # Finalize and save the figure
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "hyperparameter_distributions.png"), dpi=300)
    plt.close(fig)
