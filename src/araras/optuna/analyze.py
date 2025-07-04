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
import re
import optuna
import numpy as np
from typing import *
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PlotConfig:
    """Global configuration for matplotlib plots used in this module."""

    # Layout and sizing configurations for combined plots
    max_cols: int = 4
    numeric_subplot_size: int = 5
    box_subplot_height: int = 5
    heatmap_cell: float = 0.5
    corr_bar_min_width: int = 6
    corr_bar_scale: float = 0.6

    # Layout and sizing configurations for standalone plots
    standalone_size: Tuple[int, int] = (8, 6)

    # Font sizes for combined plots
    title_fs: int = 16
    label_fs: int = 14
    legend_fs: int = 12
    suptitle_fs: int = 24
    annotation_fs: int = 12
    bar_value_fs: int = 9
    x_tick_fs: int = 12
    y_tick_fs: int = 12

    # Font sizes for standalone plots
    standalone_title_fs: int = 20
    standalone_label_fs: int = 18
    standalone_legend_fs: int = 16
    standalone_x_tick_fs: int = 16
    standalone_y_tick_fs: int = 16

    # Padding configurations for combined plots
    title_pad: int = 10

    # Padding configurations for standalone plots
    standalone_title_pad: int = 6

    # Common label/title strings used across plots
    density_label: str = "Density"
    count_label: str = "Count"
    study_value_label: str = "Study Value"
    importance_ylabel: str = "Importance"
    importance_title: str = "Hyperparameter Importances"
    spearman_heatmap_title: str = "Spearman Correlation"
    param_corr_title: str = (
        "Parameter-Study Value Correlations\n"
        "(Red: Negative correlation = Lower values improve performance)"
    )
    param_corr_xlabel: str = "Spearman Correlation with Study Value"
    param_corr_ylabel: str = "Parameters"
    trend_suptitle: str = "Parameter-Study Value Trend Analysis"

    # Dynamic title templates
    param_title_tpl: str = "{display_name}"
    dist_standalone_title_tpl: str = "Distribution for {display_name}"
    box_standalone_title_tpl: str = "Boxplot Comparison for {display_name}"
    trend_standalone_title_tpl: str = "Trend Analysis for {display_name}"
    ranges_standalone_title_tpl: str = "Optimal Ranges for {display_name}"


PLOT_CFG = PlotConfig()
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "xtick.labelsize": PLOT_CFG.x_tick_fs,
        "ytick.labelsize": PLOT_CFG.y_tick_fs,
    }
)

# Regex used across plots to clean parameter names for titles and labels
PARAM_NAME_CLEAN_RE = re.compile(r"^params_")


def set_plot_config_param(param_name: str, value: Any) -> None:
    """Set a single parameter in :data:`PLOT_CFG`."""
    if not hasattr(PLOT_CFG, param_name):
        raise AttributeError(f"PlotConfig has no attribute {param_name!r}")

    setattr(PLOT_CFG, param_name, value)

    if param_name == "x_tick_fs":
        plt.rcParams["xtick.labelsize"] = value
    elif param_name == "y_tick_fs":
        plt.rcParams["ytick.labelsize"] = value


def set_plot_config_params(**kwargs: Any) -> None:
    """Set multiple parameters in :data:`PLOT_CFG`."""
    for name, val in kwargs.items():
        set_plot_config_param(name, val)


def format_title(template: str, display_name: str) -> str:
    """Format a title template with the given display name."""
    return template.format(display_name=display_name)


# ———————————————————————————————————————————————————————————————————————————— #
#                               Utility Functions                              #
# ———————————————————————————————————————————————————————————————————————————— #


def create_directories(
    table_dir: str, create_standalone: bool = False, save_data: bool = True
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


def get_param_display_name(param_name: str, param_name_mapping: Dict[str, str] = None) -> str:
    """Get display name for parameter, using mapping if provided."""
    if param_name_mapping and param_name in param_name_mapping:
        return param_name_mapping[param_name]
    cleaned = PARAM_NAME_CLEAN_RE.sub("", param_name)
    cleaned = cleaned.replace("_", " ")
    return cleaned.title()


def prepare_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Extract and clean completed trial data from Optuna study."""
    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    required_columns = ["state", "value"]
    available_columns = list(df.columns)
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        error_msg = (
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {available_columns}\n"
            f"This might indicate an issue with the Optuna study data.\n"
            f"Please check:\n"
            f"1. The study.db file path is correct and accessible\n"
            f"2. The study contains trials with the expected data structure\n"
            f"3. The Optuna version is compatible with this analysis code"
        )
        raise ValueError(error_msg)

    df = df.query("state == 'COMPLETE'")
    df = df.drop(columns=["number", "state"], errors="ignore")

    if df.empty:
        print("Warning: No completed trials found in the study.")
        print("Please verify that the study contains successful trials.")
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


def get_trial_subsets(df: pd.DataFrame, top_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract best and worst performing trial subsets based on loss values."""
    n_top = max(1, int(len(df) * top_frac))
    best = df.nsmallest(n_top, "loss")
    worst = df.nlargest(n_top, "loss")
    return best, worst


def format_numeric_value(x: float) -> Union[int, float, str]:
    """Format numeric values with appropriate precision for readability."""
    if pd.isna(x) or np.isinf(x):
        return x
    if abs(x - round(x)) < 1e-12:
        return int(round(x))
    if abs(x) < 1e-1:
        return f"{x:.2e}"
    return float(round(x, 2))


def save_summary_tables(
    df: pd.DataFrame,
    best: pd.DataFrame,
    worst: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    dirs: Dict[str, str],
) -> None:
    """Generate and save statistical summary tables for different trial subsets."""
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


# Adding import statements for all plot functions
#! Must be done after the utility functions to ensure no circular imports
from .describe_numeric import describe_numeric
from .create_frequency_table import create_frequency_table
from .plot_hyperparameter_distributions import plot_hyperparameter_distributions
from .plot_param_importances import plot_param_importances
from .plot_spearman_correlation import plot_spearman_correlation
from .plot_parameter_boxplots import plot_parameter_boxplots
from .plot_trend_analysis import plot_trend_analysis
from .plot_optimal_ranges_analysis import plot_optimal_ranges_analysis
from .plot_contour import plot_contour
from .plot_edf import plot_edf
from .plot_intermediate_values import plot_intermediate_values
from .plot_parallel_coordinate import plot_parallel_coordinate
from .plot_rank import plot_rank
from .plot_slice import plot_slice
from .plot_optimization_history import plot_optimization_history
from .plot_timeline import plot_timeline
from .plot_terminator_improvement import plot_terminator_improvement

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
        print(f"Study info:")
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
            print("No columns to display after applying exclusions.")
    except Exception as e:
        print(f"Error extracting study information: {str(e)}")


def analyze_study(
    study: optuna.Study,
    table_dir: str,
    top_frac: float = 0.2,
    param_name_mapping: Dict[str, str] = None,
    create_standalone: bool = False,
    save_data: bool = False,
    plots: Optional[List[str]] = None,
) -> None:
    """Comprehensive analysis of Optuna hyperparameter optimization study results.

    Args:
        study: Optuna study object containing trials to analyze.
        table_dir: Directory to save analysis results and figures.
        top_frac: Fraction of best/worst trials to analyze (default: 0.2).
        param_name_mapping: Optional mapping of parameter names to display names.
            Example: {'params_learning_rate': 'Learning Rate'}
        create_standalone: If True, generates standalone images for each plot type.
        save_data: If True, saves data for LaTeX plotting into CSV files.
        plots: List of plot types to generate. Available options:
            'distributions', 'importances', 'correlations', 'boxplots',
            'trends', 'ranges', 'contours', 'edf', 'intermediate',
            'parallel_coordinate', 'rank', 'slice', 'history', 'timeline',
            'terminator'.
            If None, generates all plots.
    """
    print("\n\nAnalyzing study...")

    # Define all available plot types
    all_plots = {
        "distributions",
        "importances",
        "correlations",
        "boxplots",
        "trends",
        "ranges",
        "contours",
        "edf",
        "intermediate",
        "parallel_coordinate",
        "rank",
        "slice",
        "history",
        "timeline",
        "terminator",
    }

    # Set plots to generate (default: all plots)
    if plots is None:
        plots_to_generate = all_plots
    else:
        plots_to_generate = set(plots)
        invalid_plots = plots_to_generate - all_plots
        if invalid_plots:
            print(f"Warning: Invalid plot types ignored: {invalid_plots}")
            plots_to_generate = plots_to_generate & all_plots

    dirs = create_directories(table_dir, create_standalone, save_data)

    df = prepare_dataframe(study)
    if df.empty:
        print("No completed trials to analyze.")
        return

    numeric_cols, categorical_cols = classify_columns(df)
    best, worst = get_trial_subsets(df, top_frac)

    print_study_columns(
        study,
        exclude=[
            "loss",
            "value",
            "number",
            "datetime_start",
            "datetime_complete",
            "duration",
            "system_attrs_completed_rung_0",
            "system_attrs_completed_rung_1",
            "system_attrs_completed_rung_2",
            "state",
        ]
        + [col for col in df.columns if col.startswith("user_")],
        param_name_mapping=param_name_mapping,
    )

    print("\nGenerating summary tables...")
    save_summary_tables(df, best, worst, numeric_cols, categorical_cols, dirs)

    if "distributions" in plots_to_generate:
        print("Generating hyperparameter distribution plots...")
        plot_hyperparameter_distributions(
            df, numeric_cols, categorical_cols, dirs, param_name_mapping, create_standalone
        )

    if "importances" in plots_to_generate:
        print("Generating parameter importances...")
        plot_param_importances(study, dirs)

    if "correlations" in plots_to_generate:
        print("Generating Spearman correlations...")
        plot_spearman_correlation(df, numeric_cols, dirs)

    if "boxplots" in plots_to_generate:
        print("Generating boxplots for parameter distributions...")
        plot_parameter_boxplots(df, best, worst, numeric_cols, dirs, param_name_mapping, create_standalone)

    if "trends" in plots_to_generate:
        print("Generating trend analysis...")
        plot_trend_analysis(df, numeric_cols, dirs, param_name_mapping, create_standalone)

    if "ranges" in plots_to_generate:
        print("Generating optimal ranges analysis...")
        plot_optimal_ranges_analysis(df, best, numeric_cols, dirs, param_name_mapping, create_standalone)

    if "contours" in plots_to_generate:
        print("Generating contour plots...")
        plot_contour(study, numeric_cols, dirs, create_standalone)

    if "edf" in plots_to_generate:
        print("Generating EDF of study values...")
        plot_edf(study, dirs)

    if "intermediate" in plots_to_generate:
        print("Generating intermediate values plots...")
        plot_intermediate_values(study, dirs)

    if "parallel_coordinate" in plots_to_generate:
        print("Generating parallel coordinate plots...")
        plot_parallel_coordinate(study, numeric_cols + categorical_cols, dirs)

    if "slice" in plots_to_generate:
        print("Generating slice plots...")
        plot_slice(study, numeric_cols + categorical_cols, dirs, create_standalone)

    if "history" in plots_to_generate:
        print("Generating optimization history plot...")
        plot_optimization_history(study, dirs)

    if "timeline" in plots_to_generate:
        print("Generating timeline plot...")
        plot_timeline(study, dirs)

    if "terminator" in plots_to_generate:
        print("Generating terminator improvement plot...")
        plot_terminator_improvement(study, dirs)
        
    if "rank" in plots_to_generate:
        print("Generating rank plots...")
        plot_rank(study, numeric_cols + categorical_cols, dirs, create_standalone)

    print(f"\nAnalysis complete! Results saved to: {table_dir}")
    print(f"- Figures: {dirs['figs']}")
    if save_data:
        print(f"- Data for LaTeX: {dirs['data']}")
        print("  * Distributions:", dirs["data_distributions"])
        print("  * Boxplots:", dirs["data_boxplots"])
        print("  * Trends:", dirs["data_trends"])
        print("  * Ranges:", dirs["data_ranges"])
        print("  * Importances:", dirs["data_importances"])
        print("  * Correlations:", dirs["data_correlations"])
    print(f"- Summary tables: {dirs['table_overall']}, {dirs['table_best']}, {dirs['table_worst']}")

    if create_standalone:
        print("- Standalone images:")
        print(f"  * Distributions: {dirs['standalone_distributions']}")
        print(f"  * Boxplots: {dirs['standalone_boxplots']}")
        print(f"  * Trends: {dirs['standalone_trends']}")
        print(f"  * Ranges: {dirs['standalone_ranges']}")
        print(f"  * Contours: {dirs['standalone_contours']}")
        print(f"  * Slices: {dirs['standalone_slices']}")
        print(f"  * Ranks: {dirs['standalone_ranks']}")

    if param_name_mapping:
        print(f"\nParameter name mappings applied:")
        for orig, display in param_name_mapping.items():
            print(f"  {orig} -> {display}")

    print(
        f"\nProcessed {len(df)} trials with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical parameters."
    )

    if plots is not None:
        print(f"Generated plots: {sorted(plots_to_generate)}")
