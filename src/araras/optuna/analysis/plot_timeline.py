import os
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import timedelta
import numpy as np
import optuna

from .analyze import PLOT_CFG


def plot_timeline(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Visualize trial durations on a timeline with detailed information."""
    considered_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    trials = [t for t in study.trials if t.state in considered_states]
    if not trials:
        print("No finished trials for timeline plot.")
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
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%H H\n%b %d, %Y')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis_date()

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
    fig.savefig(os.path.join(dirs["figs"], "study_timeline.pdf"), bbox_inches="tight")
    plt.close(fig)
