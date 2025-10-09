from typing import Dict

import matplotlib.pyplot as plt
import optuna

from araras.ml.optuna.analyzer import PLOT_CFG
from araras.ml.optuna.analysis_utils import draw_warning_box, save_plot


def plot_intermediate_values(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool = False) -> None:
    """Plot intermediate values reported during trials.

    This plot shows how reported metrics evolve during the optimization. It can
    highlight variance and potential early stopping points. An optional
    interactive Plotly version can also be saved.

    Args:
        study (optuna.Study): Optuna study containing the trials.
        dirs (Dict[str, str]): Dictionary with output directories for saving figures.
        create_plotly (bool): Whether to save an interactive HTML version of the plot.

    Returns:
        None: The figure is persisted to the directories referenced in ``dirs``.
    """
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for intermediate values plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "study_intermediate_values", "figs", create_plotly)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
    cmap = plt.get_cmap("tab10")
    for idx, t in enumerate(trials):
        if not t.intermediate_values:
            continue
        steps = sorted(t.intermediate_values.keys())
        values = [t.intermediate_values[s] for s in steps]
        color = cmap(idx % 10)
        ax.plot(
            steps,
            values,
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Step", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Intermediate Values", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_intermediate_values(study)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "study_intermediate_values", "figs", create_plotly, pfig)
    plt.close(fig)
