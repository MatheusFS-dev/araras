import os
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import optuna

from .analyze import PLOT_CFG


def plot_timeline(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """Visualize trial durations on a timeline."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("No completed trials for timeline plot.")
        return

    trials.sort(key=lambda t: t.number)
    starts = [t.datetime_start for t in trials]
    ends = [t.datetime_complete for t in trials]
    numbers = [t.number for t in trials]

    start_nums = mdates.date2num(starts)
    end_nums = mdates.date2num(ends)
    durations = end_nums - start_nums

    fig, ax = plt.subplots(figsize=PLOT_CFG.importance_size)
    ax.barh(numbers, durations, left=start_nums, height=0.8, color="skyblue", edgecolor="black")

    ax.xaxis_date()
    ax.set_xlabel("Time", fontsize=PLOT_CFG.label_fs)
    ax.set_ylabel("Trial", fontsize=PLOT_CFG.label_fs)
    ax.set_title("Timeline", pad=PLOT_CFG.title_pad)
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(dirs["figs"], "timeline.pdf"), bbox_inches="tight")
    plt.close(fig)
