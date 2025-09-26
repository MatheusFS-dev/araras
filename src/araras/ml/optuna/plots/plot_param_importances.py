from araras.core import *

import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances

from araras.ml.optuna.analyzer import PLOT_CFG
from araras.ml.optuna.analysis_utils import (
    save_data_for_latex,
    get_param_display_name,
    draw_warning_box,
    save_plot,
)


def plot_param_importances(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool = False) -> None:
    """Generate bar charts and CSV summaries of Optuna parameter importances.

    Args:
        study: Optuna study containing the optimisation history.
        dirs: Mapping of directory identifiers to output folders.
        create_plotly: Whether to export an interactive HTML bar chart in
            addition to the static Matplotlib figure.

    Returns:
        None: CSV data files and plots are saved under the paths provided in
        ``dirs``.
    """
    # Calculate parameter importances using Optuna's algorithm
    importances = get_param_importances(study)

    # Convert to DataFrame and sort by importance (descending)
    df_imp = pd.DataFrame(list(importances.items()), columns=["Parameter", "Importance"]).sort_values(
        "Importance", ascending=False
    )
    if df_imp.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No importances could be computed.")
        ax.set_title(
            PLOT_CFG.importance_title,
            pad=PLOT_CFG.title_pad,
            fontsize=PLOT_CFG.standalone_title_fs,
        )
        plt.tight_layout()
        save_plot(fig, dirs, "params_importances", "figs", create_plotly)
        plt.close(fig)
        return

    # Save data for LaTeX
    save_data_for_latex(
        {
            "parameter": df_imp["Parameter"].tolist(),
            "importance": df_imp["Importance"].tolist(),
        },
        "param_importances",
        dirs["data_importances"],
    )

    # Create horizontal bar chart visualization
    plt.figure(figsize=PLOT_CFG.standalone_size)
    display_names = [get_param_display_name(p) for p in df_imp["Parameter"]]
    bars = plt.barh(range(len(df_imp)), df_imp["Importance"])
    plt.yticks(range(len(df_imp)), display_names, fontsize=PLOT_CFG.y_tick_fs)
    plt.xlabel(
        PLOT_CFG.importance_ylabel, fontsize=PLOT_CFG.standalone_label_fs
    )  # Importance score on x-axis
    plt.title(
        PLOT_CFG.importance_title,
        pad=PLOT_CFG.title_pad,
        fontsize=PLOT_CFG.standalone_title_fs,
    )
    for bar, importance in zip(bars, df_imp["Importance"]):
        width = bar.get_width()
        plt.text(
            width + PLOT_CFG.bar_value_offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"{importance:.3f}",
            ha="left",
            va="center",
            fontsize=PLOT_CFG.bar_value_fs,
            bbox=dict(
                facecolor="white",
                alpha=0,
                edgecolor="none",
                pad=PLOT_CFG.bar_value_pad,
            ),
        )
    xmax = df_imp["Importance"].max()
    pad = max(0.05, xmax * 0.1)
    plt.xlim(0, xmax + pad)
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    # Save with high resolution
    fig = plt.gcf()
    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_param_importances(study)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "params_importances", "figs", create_plotly, pfig)
    plt.close()  # Close figure to free memory
