"""
Utility functions for :mod:`araras.ml.optuna` analysis tools.
"""

from __future__ import annotations

from araras.core import *

import os
import re
import math
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from optuna.terminator import BaseImprovementEvaluator, RegretBoundEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS

from .analyzer import PLOT_CFG


# Regex used across plots to clean parameter names for titles and labels
PARAM_NAME_CLEAN_RE = re.compile(r"^params_")


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_trial_subsets(df: pd.DataFrame, top_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract best and worst performing trial subsets.

    Args:
        df: DataFrame with a ``loss`` column.
        top_frac: Fraction of top trials to extract.

    Returns:
        Tuple of DataFrames ``(best, worst)``.
    """
    n_top = max(1, int(len(df) * top_frac))
    best = df.nsmallest(n_top, "loss")
    worst = df.nlargest(n_top, "loss")
    return best, worst


def format_numeric_value(x: float) -> Union[int, float, str]:
    """Format numeric values for readability.

    Args:
        x: Numeric value to format.

    Returns:
        Formatted value with reduced precision when possible.
    """
    if pd.isna(x) or np.isinf(x):
        return x
    if abs(x - round(x)) < 1e-12:
        return int(round(x))
    if abs(x) < 1e-1:
        return f"{x:.2e}"
    return float(round(x, 2))


def describe_numeric(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Generate descriptive statistics for numeric hyperparameters.

    Args:
        data: DataFrame with numeric columns.
        cols: Target column names.

    Returns:
        DataFrame with summary statistics.
    """
    stats = []
    for col in cols:
        arr = data[col].dropna()
        if arr.empty:
            formatted = {
                "Parameter": col,
                "Mean": "No data",
                "Std": "No data",
                "Median": "No data",
                "Min (25% quantile)": "No data",
                "Max (75% quantile)": "No data",
                "Min (5% quantile)": "No data",
                "Max (95% quantile)": "No data",
            }
        else:
            raw = {
                "Parameter": col,
                "Mean": arr.mean(),
                "Std": arr.std(),
                "Median": arr.median(),
                "Min (25% quantile)": arr.quantile(0.25),
                "Max (75% quantile)": arr.quantile(0.75),
                "Min (5% quantile)": arr.quantile(0.05),
                "Max (95% quantile)": arr.quantile(0.95),
            }
            formatted = {"Parameter": col}
            for k, v in raw.items():
                if k != "Parameter":
                    formatted[k] = format_numeric_value(v)
        stats.append(formatted)
    return pd.DataFrame(stats)


def create_frequency_table(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Generate frequency tables for categorical hyperparameters.

    Args:
        data: DataFrame containing the data.
        cols: Categorical column names.

    Returns:
        DataFrame with category counts and fractions.
    """
    rows = []
    for col in cols:
        counts = data[col].value_counts(normalize=True)
        for cat, frac in counts.items():
            rows.append({
                "Parameter": col,
                "Category": cat,
                "Fraction": round(frac, 4),
                "Count": int(data[col].value_counts()[cat]),
            })
    return pd.DataFrame(rows)


def _safe_plot(plot_name: str, func: Callable, *args: Any, **kwargs: Any) -> None:
    """Execute a plotting function and catch errors.

    Args:
        plot_name: Name of the plot being generated.
        func: Plotting callable.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.

    Returns:
        None

    Notes:
        Exceptions are logged and printed; they are not raised.
    """
    try:
        func(*args, **kwargs)
    except Exception as e:  # pragma: no cover - user feedback only
        logger_error.error(f"{RED}Error generating {plot_name} plot: {e}{RESET}")
        traceback.print_exc()


def format_title(template: str, display_name: str) -> str:
    """Format a title template with a display name."""

    return template.format(display_name=display_name)


def calculate_grid(
    n_plots: int,
    subplot_width: int,
    subplot_height: int,
    base_max_cols: int,
) -> Tuple[int, int]:
    """Calculate grid dimensions for subplots ensuring image size limits.

    Args:
        n_plots: Number of subplots desired.
        subplot_width: Width of each subplot in inches.
        subplot_height: Height of each subplot in inches.
        base_max_cols: Desired maximum columns before adjustment.

    Returns:
        Tuple ``(n_rows, n_cols)`` for ``plt.subplots``.
    """
    if n_plots <= 0:
        return 0, 0

    dpi = plt.rcParams.get("figure.dpi", 100)
    max_px = (2 ** 16) - 1

    max_cols_by_width = max_px // int(subplot_width * dpi)
    max_rows_by_height = max_px // int(subplot_height * dpi)

    max_cols_by_width = max(1, max_cols_by_width)
    max_rows_by_height = max(1, max_rows_by_height)

    n_cols = min(base_max_cols, max_cols_by_width, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    if n_rows > max_rows_by_height:
        n_cols = min(max_cols_by_width, math.ceil(n_plots / max_rows_by_height))
        n_rows = math.ceil(n_plots / n_cols)

    return n_rows, n_cols


def draw_warning_box(ax: plt.Axes, message: str) -> None:
    """Display a warning message in a plot area."""

    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=PLOT_CFG.label_fs,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        linespacing=1.5,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def create_directories(
    table_dir: str,
    create_standalone: bool = False,
    save_data: bool = True,
    create_plotly: bool = False,
) -> Dict[str, str]:
    """Create organized subdirectories for storing analysis outputs."""
    dirs = {
        "figs": os.path.join(table_dir, "figures"),
        "table_best": os.path.join(table_dir, "best"),
        "table_worst": os.path.join(table_dir, "worst"),
        "table_overall": os.path.join(table_dir, "overall"),
    }

    if save_data:
        dirs.update(
            {
                "data": os.path.join(table_dir, "data"),
                "data_distributions": os.path.join(table_dir, "data", "distributions"),
                "data_boxplots": os.path.join(table_dir, "data", "boxplots"),
                "data_trends": os.path.join(table_dir, "data", "trends"),
                "data_ranges": os.path.join(table_dir, "data", "ranges"),
                "data_importances": os.path.join(table_dir, "data", "importances"),
                "data_correlations": os.path.join(table_dir, "data", "correlations"),
            }
        )
    else:
        dirs.update(
            {
                "data": None,
                "data_distributions": None,
                "data_boxplots": None,
                "data_trends": None,
                "data_ranges": None,
                "data_importances": None,
                "data_correlations": None,
            }
        )

    if create_standalone:
        dirs.update(
            {
                "standalone_distributions": os.path.join(table_dir, "figures", "standalone", "distributions"),
                "standalone_boxplots": os.path.join(table_dir, "figures", "standalone", "boxplots"),
                "standalone_trends": os.path.join(table_dir, "figures", "standalone", "trends"),
                "standalone_ranges": os.path.join(table_dir, "figures", "standalone", "ranges"),
                "standalone_contours": os.path.join(table_dir, "figures", "standalone", "contours"),
                "standalone_slices": os.path.join(table_dir, "figures", "standalone", "slices"),
                "standalone_ranks": os.path.join(table_dir, "figures", "standalone", "ranks"),
            }
        )

    if create_plotly:
        dirs.update({"plotly": os.path.join(table_dir, "plotly")})
        if create_standalone:
            dirs.update(
                {
                    "plotly_standalone_distributions": os.path.join(table_dir, "plotly", "standalone", "distributions"),
                    "plotly_standalone_boxplots": os.path.join(table_dir, "plotly", "standalone", "boxplots"),
                    "plotly_standalone_trends": os.path.join(table_dir, "plotly", "standalone", "trends"),
                    "plotly_standalone_ranges": os.path.join(table_dir, "plotly", "standalone", "ranges"),
                    "plotly_standalone_contours": os.path.join(table_dir, "plotly", "standalone", "contours"),
                    "plotly_standalone_slices": os.path.join(table_dir, "plotly", "standalone", "slices"),
                    "plotly_standalone_ranks": os.path.join(table_dir, "plotly", "standalone", "ranks"),
                }
            )

    for dir_path in dirs.values():
        if dir_path is not None:
            os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_data_for_latex(data_dict: Dict[str, Any], filename: str, data_dir: str) -> None:
    """Save graph data to CSV files for LaTeX plotting."""
    if data_dir is None:
        return
    filepath = os.path.join(data_dir, f"{filename}.csv")
    df = pd.DataFrame(data_dict)
    df.to_csv(filepath, index=False)


def save_plotly_html(fig: Any, filepath: str) -> None:
    """Save a Plotly figure to an HTML file."""
    try:
        import plotly.io as pio

        pio.write_html(fig, filepath, include_plotlyjs="cdn")
    except Exception as e:  # pragma: no cover - runtime warning only
        logger_error.error(f"{RED}Error saving plotly figure {filepath}: {e}{RESET}")


def save_plot(
    fig: plt.Figure,
    dirs: Dict[str, str],
    base_name: str,
    subdir_key: str,
    create_plotly: bool,
    plotly_fig: Any = None,
) -> None:
    """Save Matplotlib figure and optionally a Plotly HTML version."""
    pdf_path = os.path.join(dirs[subdir_key], f"{base_name}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    if create_plotly and plotly_fig is not None:
        plotly_key = "plotly" if subdir_key == "figs" else f"plotly_{subdir_key}"
        html_dir = dirs.get(plotly_key)
        if html_dir:
            html_path = os.path.join(html_dir, f"{base_name}.html")
            save_plotly_html(plotly_fig, html_path)


def get_param_display_name(param_name: str, param_name_mapping: Dict[str, str] | None = None) -> str:
    """Return a display-friendly name for a hyperparameter."""

    if param_name_mapping and param_name in param_name_mapping:
        return param_name_mapping[param_name]
    cleaned = PARAM_NAME_CLEAN_RE.sub("", param_name)
    cleaned = cleaned.replace("_", " ")
    return cleaned.title()


def prepare_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Extract and clean completed trial data from an Optuna study."""

    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    required_columns = ["state", "value"]
    available_columns = list(df.columns)
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        error_msg = (
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {available_columns}\n"
            f"This might indicate an issue with the Optuna study data.\n"
            "Please check:\n"
            "1. The study.db file path is correct and accessible\n"
            "2. The study contains trials with the expected data structure\n"
            "3. The Optuna version is compatible with this analysis code"
        )
        raise ValueError(error_msg)

    df = df.query("state == 'COMPLETE'")
    df = df.drop(columns=["number", "state"], errors="ignore")

    if df.empty:
        return df

    df = df.rename(columns={"value": "loss"})
    finite = df["loss"].replace([np.inf, -np.inf], np.nan)
    worst = finite.max()
    df["loss"] = df["loss"].replace([np.inf, -np.inf], worst).fillna(worst)
    return df


def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split DataFrame columns into numeric and categorical parameter types."""

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "loss"]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols


def save_summary_tables(
    df: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    dirs: Dict[str, str],
) -> None:
    """Generate and save statistical summary tables."""
    datasets = [("overall", df), ("best", best), ("worst", worst)]
    for label, subset in datasets:
        dir_key = f"table_{label}"
        target_dir = dirs[dir_key]
        describe_numeric(subset, numeric_cols).to_csv(
            os.path.join(target_dir, f"{label}_numeric_summary.csv"), index=False
        )
        create_frequency_table(subset, categorical_cols).to_csv(
            os.path.join(target_dir, f"{label}_categorical_frequencies.csv"), index=False
        )


def print_study_columns(
    study: optuna.Study,
    exclude: Optional[List[str]] = None,
    param_name_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """Print the names of the DataFrame columns from the study as a bullet list."""
    if exclude is None:
        exclude = []
    try:
        df = study.trials_dataframe()
        all_columns = list(df.columns)
        filtered_columns = [col for col in all_columns if col not in exclude]
        print("-" * 50)
        print("Study info:")
        print(f"• Total trials: {len(df)}")
        if "state" in df.columns:
            state_counts = df["state"].value_counts()
            for state, count in state_counts.items():
                print(f"• {state} trials: {count}")
        if filtered_columns:
            print("Parameter Template:")
            print("{")
            for col in filtered_columns:
                if col.startswith("params_"):
                    param_name = col
                    if param_name_mapping:
                        display_name = param_name_mapping.get(param_name, param_name)
                        print(f'    "{param_name}": "{display_name}",')
                    else:
                        print(f'    "{param_name}": "{param_name}",')
            print("}")
            print("-" * 50)
        else:
            print(f"{ORANGE}No columns to display after applying exclusions.{RESET}")
    except Exception as e:
        logger_error.error(f"{RED}Error extracting study information: {str(e)}{RESET}")


def analyze_improvement_variance(
    study: optuna.Study,
    window_size: int = 10,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
) -> List[float]:
    """Compute variance of improvement across sliding windows.

    This helper replicates the improvement calculation performed by
    :class:`ImprovementStagnation` without altering the study. It is intended
    as a diagnostic tool to inspect how the improvement variance evolves over
    time so that a suitable ``variance_threshold`` can be chosen.

    Args:
        study: The Optuna study containing trials to analyse.
        window_size: Number of recent improvements used for each variance
            calculation.
        min_n_trials: Minimum number of completed trials required before
            variance values are produced.
        improvement_evaluator: Custom evaluator used to estimate potential
            improvement. Defaults to :class:`RegretBoundEvaluator`.

    Returns:
        List[float]: Sequence of variance values calculated after each
        completed trial. Elements corresponding to trials before the warm-up
        period contain ``numpy.nan``.

    Raises:
        ValueError: If ``window_size`` is less than ``1`` or ``min_n_trials`` is
            negative.

    Notes:
        The function prints the trial indices associated with each computed
        variance for convenience.
    """

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if min_n_trials < 0:
        raise ValueError("min_n_trials must be non-negative")

    if improvement_evaluator is None:
        improvement_evaluator = RegretBoundEvaluator()

    completed_trials: List[optuna.trial.FrozenTrial] = []
    improvements: List[float] = []
    trial_numbers: List[int] = []
    variances: List[float] = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        completed_trials.append(trial)
        trial_numbers.append(trial.number)

        improvement = improvement_evaluator.evaluate(
            trials=completed_trials,
            study_direction=study.direction,
        )
        improvements.append(improvement)

        if len(completed_trials) >= max(min_n_trials, window_size):
            window_improvements = improvements[-window_size:]
            variance = float(np.var(window_improvements))
            print(
                f"Variance of trials {trial_numbers[-window_size:]}: {variance:.3e}"
            )
            variances.append(variance)
        else:
            variances.append(float('nan'))

    return variances

