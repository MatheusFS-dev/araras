
from araras.core import *

import optuna
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ———————————————————————————————————————————————————————————————————————————— #
#                                 Configuration                                #
# ———————————————————————————————————————————————————————————————————————————— #


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
    bar_value_fs: int = 12
    bar_value_pad: float = 0.2
    bar_value_offset: float = 0.01
    heatmap_value_fs: int = 8
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
    param_corr_title: str = "Parameter-Study Value Correlations"
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

from .analysis_utils import (
    _safe_plot,
    classify_columns,
    create_directories,
    get_trial_subsets,
    prepare_dataframe,
    print_study_columns,
    save_summary_tables,
)


def set_plot_config_param(param_name: str, value: Any) -> None:
    """Set a single parameter in :data:`PLOT_CFG`."""
    if not hasattr(PLOT_CFG, param_name):
        raise AttributeError(f"PlotConfig has no attribute {param_name!r}")

    setattr(PLOT_CFG, param_name, value)

    if param_name == "x_tick_fs":
        plt.rcParams["xtick.labelsize"] = value
    elif param_name == "y_tick_fs":
        plt.rcParams["ytick.labelsize"] = value


# Set multiple PlotConfig parameters at once
def set_plot_config_params(**kwargs: Any) -> None:
    """Set multiple parameters in :data:`PLOT_CFG`."""

    for name, val in kwargs.items():
        set_plot_config_param(name, val)


# Adding import statements for all plot functions
#! Must be done after the utility functions to ensure no circular imports
from .plots.plot_hyperparameter_distributions import plot_hyperparameter_distributions
from .plots.plot_param_importances import plot_param_importances
from .plots.plot_spearman_correlation import plot_spearman_correlation
from .plots.plot_parameter_boxplots import plot_parameter_boxplots
from .plots.plot_trend_analysis import plot_trend_analysis
from .plots.plot_optimal_ranges_analysis import plot_optimal_ranges_analysis
from .plots.plot_contour import plot_contour
from .plots.plot_edf import plot_edf
from .plots.plot_intermediate_values import plot_intermediate_values
from .plots.plot_parallel_coordinate import plot_parallel_coordinate
from .plots.plot_rank import plot_rank
from .plots.plot_slice import plot_slice
from .plots.plot_optimization_history import plot_optimization_history
from .plots.plot_timeline import plot_timeline
from .plots.plot_terminator_improvement import plot_terminator_improvement

# ———————————————————————————————————————————————————————————————————————————— #


def analyze_study(
    study: optuna.Study,
    table_dir: str,
    top_frac: float = 0.2,
    param_name_mapping: Dict[str, str] = None,
    create_standalone: bool = False,
    save_data: bool = False,
    create_plotly: bool = False,
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
        create_plotly: If True, also saves interactive Plotly HTML versions of the figures.
        plots: List of plot types to generate. Available options:
            'distributions', 'importances', 'correlations', 'boxplots',
            'trends', 'ranges', 'contours', 'edf', 'intermediate',
            'parallel_coordinate', 'slice', 'rank', 'history', 'timeline',
            'terminator'.
            Deactivated by default:
                - 'parallel_coordinate' (Too much of a mess to be useful)
                - 'rank' (Can cause crashes and not very useful)
                - 'contours' (For many parameters, this is quite messy)
                - 'slice' (Similar to trend)
                - 'edf' (Useful, but not always needed)
                - 'intermediate' (Similar to history. Did the study converge?)
            If None, generates a recommended default set of plots.
    """

    # Define all available plot types in order
    valid_plots = [
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
    ]

    # Default plots exclude the noisy/experimental ones
    default_plots = [
        p
        for p in valid_plots
        if p
        not in {
            "parallel_coordinate",
            "rank",
            "contours",
            "slice",
            "edf",
            "intermediate",
        }
    ]

    # Validate requested plot names
    if plots is None:
        plots_to_generate = default_plots
    else:
        invalid_plots = [p for p in plots if p not in valid_plots]
        if invalid_plots:
            logger.warning(f"{YELLOW}Invalid plot types ignored: {invalid_plots}{RESET}")
        plots_to_generate = [p for p in plots if p in valid_plots]

    dirs = create_directories(table_dir, create_standalone, save_data, create_plotly)

    df = prepare_dataframe(study)
    if df.empty:
        logger_error.error(f"{RED}No completed trials found in the study.{RESET}")
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

    print(f"\n{BLUE}{BOLD}Analyzing study...{RESET}")
    print("     Generating summary tables...")
    save_summary_tables(df, best, worst, numeric_cols, categorical_cols, dirs)

    plot_actions = {
        "distributions": (
            "hyperparameter distribution plots",
            lambda: _safe_plot(
                "distributions",
                plot_hyperparameter_distributions,
                df,
                numeric_cols,
                categorical_cols,
                dirs,
                param_name_mapping,
                create_standalone,
                create_plotly,
            ),
        ),
        "importances": (
            "parameter importances",
            lambda: _safe_plot("importances", plot_param_importances, study, dirs, create_plotly),
        ),
        "correlations": (
            "Spearman correlations",
            lambda: _safe_plot(
                "correlations", plot_spearman_correlation, df, numeric_cols, dirs, create_plotly
            ),
        ),
        "boxplots": (
            "boxplots for parameter distributions",
            lambda: _safe_plot(
                "boxplots",
                plot_parameter_boxplots,
                df,
                best,
                worst,
                numeric_cols,
                dirs,
                param_name_mapping,
                create_standalone,
                create_plotly,
            ),
        ),
        "trends": (
            "trend analysis",
            lambda: _safe_plot(
                "trends",
                plot_trend_analysis,
                df,
                numeric_cols,
                dirs,
                param_name_mapping,
                create_standalone,
                create_plotly,
            ),
        ),
        "ranges": (
            "optimal ranges analysis",
            lambda: _safe_plot(
                "ranges",
                plot_optimal_ranges_analysis,
                df,
                best,
                numeric_cols,
                dirs,
                param_name_mapping,
                create_standalone,
                create_plotly,
            ),
        ),
        "contours": (
            "contour plots",
            lambda: _safe_plot(
                "contours",
                plot_contour,
                study,
                numeric_cols,
                dirs,
                create_standalone,
                create_plotly,
            ),
        ),
        "edf": (
            "EDF of study values",
            lambda: _safe_plot("edf", plot_edf, study, dirs, create_plotly),
        ),
        "intermediate": (
            "intermediate values plots",
            lambda: _safe_plot("intermediate", plot_intermediate_values, study, dirs, create_plotly),
        ),
        "parallel_coordinate": (
            "parallel coordinate plots",
            lambda: _safe_plot(
                "parallel_coordinate",
                plot_parallel_coordinate,
                study,
                numeric_cols + categorical_cols,
                dirs,
                create_plotly,
            ),
        ),
        "slice": (
            "slice plots",
            lambda: _safe_plot(
                "slice",
                plot_slice,
                study,
                numeric_cols + categorical_cols,
                dirs,
                create_standalone,
                create_plotly,
            ),
        ),
        "history": (
            "optimization history plot",
            lambda: _safe_plot("history", plot_optimization_history, study, dirs, create_plotly),
        ),
        "timeline": (
            "timeline plot",
            lambda: _safe_plot("timeline", plot_timeline, study, dirs, create_plotly),
        ),
        "terminator": (
            "terminator improvement plot",
            lambda: _safe_plot("terminator", plot_terminator_improvement, study, dirs, create_plotly),
        ),
        #! Deprecated
        "rank": (
            "rank plots",
            lambda: _safe_plot(
                "rank",
                plot_rank,
                study,
                numeric_cols + categorical_cols,
                dirs,
                create_standalone,
                create_plotly,
            ),
        ),
    }

    for plot_name in plots_to_generate:
        desc, action = plot_actions.get(plot_name, (None, None))
        if action is not None:
            print(f"     Generating {desc}...")
            action()

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

    if create_plotly:
        print("- Plotly HTML files:")
        print(f"  * Combined: {dirs['plotly']}")
        if create_standalone:
            print(f"  * Standalone Distributions: {dirs['plotly_standalone_distributions']}")
            print(f"  * Standalone Boxplots: {dirs['plotly_standalone_boxplots']}")
            print(f"  * Standalone Trends: {dirs['plotly_standalone_trends']}")
            print(f"  * Standalone Ranges: {dirs['plotly_standalone_ranges']}")
            print(f"  * Standalone Contours: {dirs['plotly_standalone_contours']}")
            print(f"  * Standalone Slices: {dirs['plotly_standalone_slices']}")

    if param_name_mapping:
        print(f"\nParameter name mappings applied:")
        for orig, display in param_name_mapping.items():
            print(f"  {orig} -> {display}")

    print(
        f"\nProcessed {len(df)} trials with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical parameters."
    )

    if plots is not None:
        print(f"Generated plots: {sorted(plots_to_generate)}")
