from araras.core import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import timedelta
import numpy as np
import optuna

from araras.ml.optuna.analyzer import PLOT_CFG
from araras.ml.optuna.analysis_utils import draw_warning_box, save_plot

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"set_ticklabels\(\) should only be used with a fixed number of ticks.*",
    category=UserWarning,
)


def plot_timeline(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool = False) -> None:
    """Visualize trial durations on a timeline with detailed information.

    The figure shows when each trial started and how long it ran. Failed and
    pruned trials are displayed alongside completed ones. An optional Plotly
    version can be generated for interactive exploration.

    Args:
        study (optuna.Study): Optuna study with the recorded trials.
        dirs (Dict[str, str]): Dictionary with output directories for saving figures.
        create_plotly (bool): Whether to save an interactive HTML version of the plot.

    Returns:
        None: The timeline visualisation is saved within the directories
        referenced by ``dirs``.
    """
    considered_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    trials = [t for t in study.trials if t.state in considered_states]
    if not trials:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No finished trials for timeline plot.")
        plt.tight_layout()
        save_plot(fig, dirs, "study_timeline", "figs", create_plotly)
        plt.close(fig)
        return

    trials.sort(key=lambda t: t.number)
    starts = [t.datetime_start for t in trials]
    ends = [t.datetime_complete for t in trials]
    numbers = [t.number for t in trials]
    states = [t.state for t in trials]

    start_nums = mdates.date2num(starts)
    end_nums = mdates.date2num(ends)
    durations = end_nums - start_nums
    durations_sec = [(e - s).total_seconds() for s, e in zip(starts, ends)]

    colors = []
    for s in states:
        if s == optuna.trial.TrialState.COMPLETE:
            colors.append("green")
        elif s == optuna.trial.TrialState.PRUNED:
            colors.append("orange")
        else:
            colors.append("red")

    fig_width = PLOT_CFG.standalone_size[0]
    fig_height = PLOT_CFG.standalone_size[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.barh(numbers, durations, left=start_nums, height=1.0, color=colors)
    ax.set_ylim(min(numbers) - 1, max(numbers) + 1)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis_date()

    # Calculate averages by state
    complete_durations = [d for d, s in zip(durations_sec, states) if s == optuna.trial.TrialState.COMPLETE]
    pruned_durations = [d for d, s in zip(durations_sec, states) if s == optuna.trial.TrialState.PRUNED]
    fail_durations = [d for d, s in zip(durations_sec, states) if s == optuna.trial.TrialState.FAIL]

    # Format average text with all metrics
    avg_lines = []

    # Total study duration (from first start to last completion)
    total_study_duration = (max(ends) - min(starts)).total_seconds()
    avg_lines.append(f"Total study duration: {timedelta(seconds=int(total_study_duration))}\n")

    # Total average
    avg_total = float(np.mean(durations_sec))
    avg_lines.append(f"Avg duration [TOTAL]: {timedelta(seconds=int(avg_total))}")

    # Complete trials average
    if complete_durations:
        avg_complete = float(np.mean(complete_durations))
        avg_lines.append(f"Avg duration [COMPLETE]: {timedelta(seconds=int(avg_complete))}")

    # Pruned trials average
    if pruned_durations:
        avg_pruned = float(np.mean(pruned_durations))
        avg_lines.append(f"Avg duration [PRUNED]: {timedelta(seconds=int(avg_pruned))}")

    # Failed trials average
    if fail_durations:
        avg_fail = float(np.mean(fail_durations))
        avg_lines.append(f"Avg duration [FAIL]: {timedelta(seconds=int(avg_fail))}")

    avg_text = "\n".join(avg_lines)

    ax.set_xlabel("Time", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel("Trial", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title("Timeline", pad=PLOT_CFG.title_pad, fontsize=PLOT_CFG.standalone_title_fs)
    tick_dates = mdates.num2date(ax.get_xticks())
    labels = []
    prev_date = None
    for d in tick_dates:
        time_str = d.strftime("%H:%M")
        date_str = d.strftime("%b %d, %Y")
        if prev_date == date_str:
            labels.append(time_str)
        else:
            labels.append(time_str + "\n" + date_str)
            prev_date = date_str
    ax.set_xticklabels(labels)
    # fig.autofmt_xdate()

    legend_elems = [
        Patch(facecolor="green", label="COMPLETE"),
        Patch(facecolor="orange", label="PRUNED"),
        Patch(facecolor="red", label="FAIL"),
    ]
    ax.legend(handles=legend_elems, fontsize=PLOT_CFG.legend_fs)

    ax.text(
        0.05,
        0.95,
        avg_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=PLOT_CFG.annotation_fs,
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pfig = None
    if create_plotly:
        import optuna.visualization as ov

        try:
            pfig = ov.plot_timeline(study)
        except Exception:
            pfig = None

    save_plot(fig, dirs, "study_timeline", "figs", create_plotly, pfig)
    plt.close(fig)
