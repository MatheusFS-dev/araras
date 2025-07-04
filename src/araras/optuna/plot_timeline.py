import os
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    colors = []
    for s in states:
        if s == optuna.trial.TrialState.COMPLETE:
            colors.append("green")
        elif s == optuna.trial.TrialState.PRUNED:
            colors.append("orange")
        else:
            colors.append("red")

    fig_width = PLOT_CFG.importance_size[0] * 1.5
    fig_height = PLOT_CFG.importance_size[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.barh(numbers, durations, left=start_nums, height=0.8, color=colors, edgecolor="black")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.xaxis_date()

    avg_duration = np.mean(durations) * 24 * 3600
    avg_text = f"Avg duration: {timedelta(seconds=int(avg_duration))}"

    ax.set_xlabel("Time", fontsize=PLOT_CFG.label_fs)
    ax.set_ylabel("Trial", fontsize=PLOT_CFG.label_fs)
    ax.set_title("Timeline", pad=PLOT_CFG.title_pad)
    fig.autofmt_xdate()

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="green", edgecolor="black", label="COMPLETE"),
        Patch(facecolor="orange", edgecolor="black", label="PRUNED"),
        Patch(facecolor="red", edgecolor="black", label="FAIL"),
    ]
    ax.legend(handles=legend_elems, fontsize=PLOT_CFG.legend_fs)

    ax.text(
        0.01,
        0.99,
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
