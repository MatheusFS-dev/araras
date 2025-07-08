import os
from typing import Dict, List, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna

from optuna.terminator import (
    BaseErrorEvaluator,
    BaseImprovementEvaluator,
    CrossValidationErrorEvaluator,
    RegretBoundEvaluator,
)
from optuna.terminator.erroreval import StaticErrorEvaluator
from optuna.terminator.improvement.evaluator import (
    BestValueStagnationEvaluator,
    DEFAULT_MIN_N_TRIALS,
)

from .analyze import PLOT_CFG


PADDING_RATIO_Y = 0.05
OPACITY = 0.25


class _ImprovementInfo(NamedTuple):
    trial_numbers: List[int]
    improvements: List[float]
    errors: Optional[List[float]]


def _get_improvement_info(
    study: optuna.Study,
    get_error: bool = False,
    improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    error_evaluator: Optional[BaseErrorEvaluator] = None,
) -> _ImprovementInfo:
    if study._is_multi_objective():
        raise ValueError("This function does not support multi-objective optimization study.")

    if improvement_evaluator is None:
        improvement_evaluator = RegretBoundEvaluator()
    if error_evaluator is None:
        if isinstance(improvement_evaluator, BestValueStagnationEvaluator):
            error_evaluator = StaticErrorEvaluator(constant=0)
        else:
            error_evaluator = CrossValidationErrorEvaluator()

    trial_numbers: List[int] = []
    completed_trials: List[optuna.trial.FrozenTrial] = []
    improvements: List[float] = []
    errors: List[float] = []

    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_trials.append(trial)

        if len(completed_trials) == 0:
            continue

        trial_numbers.append(trial.number)

        improvement = improvement_evaluator.evaluate(
            trials=completed_trials,
            study_direction=study.direction,
        )
        improvements.append(improvement)

        if get_error:
            error = error_evaluator.evaluate(
                trials=completed_trials,
                study_direction=study.direction,
            )
            errors.append(error)

    if len(errors) == 0:
        errors_list = None
    else:
        errors_list = errors

    return _ImprovementInfo(trial_numbers, improvements, errors_list)


def _get_y_range(info: _ImprovementInfo, min_n_trials: int) -> tuple[float, float]:
    min_value = min(info.improvements)
    if info.errors is not None:
        min_value = min(min_value, min(info.errors))

    if len(info.trial_numbers) > min_n_trials:
        max_value = max(info.improvements[min_n_trials:])
    else:
        max_value = max(info.improvements)

    if info.errors is not None:
        max_value = max(max_value, max(info.errors))

    padding = (max_value - min_value) * PADDING_RATIO_Y
    return min_value - padding, max_value + padding


def plot_terminator_improvement(
    study: optuna.Study,
    dirs: Dict[str, str],
    plot_error: bool = True,
    print_variance: bool = False,
    improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    error_evaluator: Optional[BaseErrorEvaluator] = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> None:
    """Plot the potentials for future objective improvement using Matplotlib."""

    info = _get_improvement_info(
        study,
        get_error=plot_error,
        improvement_evaluator=improvement_evaluator,
        error_evaluator=error_evaluator,
    )

    if not info.trial_numbers:
        print("No completed trials for terminator improvement plot.")
        return

    # Calculate and print variance of last 5 trials after warm-up phase
    if print_variance:
        if len(info.trial_numbers) > min_n_trials + 5:
            for i in range(min_n_trials, len(info.trial_numbers) - 4):
                last_5_trials = info.improvements[i : i + 5]
                variance = np.var(last_5_trials)
                print(f"Variance of trials {info.trial_numbers[i:i + 5]}: {variance}")

    fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)

    # Plot improvement until min_n_trials with lighter color
    ax.plot(
        info.trial_numbers[: min_n_trials + 1],
        info.improvements[: min_n_trials + 1],
        color="blue",
        label="Improvement Curve\n(Room left to improve)",
        linewidth=2,
    )

    if len(info.trial_numbers) > min_n_trials:
        ax.plot(
            info.trial_numbers[min_n_trials:],
            info.improvements[min_n_trials:],
            color="blue",
            linewidth=2,
        )

    if plot_error and info.errors is not None:
        ax.plot(
            info.trial_numbers,
            info.errors,
            color="red",
            label="Error Curve\n(Improvement due to noise)",
            linewidth=2,
        )

    ax.axvspan(
        min(info.trial_numbers),
        min_n_trials,
        color="gray",
        label=f"Warm-up Phase ({min_n_trials} trials)",
        alpha=OPACITY,
    )

    ymin, ymax = _get_y_range(info, min_n_trials)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Trial", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel("Improvement", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title(
        "Improvement vs Trials Plot",
        pad=PLOT_CFG.title_pad,
        fontsize=PLOT_CFG.standalone_title_fs,
    )
    ax.legend(fontsize=PLOT_CFG.legend_fs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(os.path.join(dirs["figs"], "terminator_improvement.pdf"), bbox_inches="tight")
    plt.close(fig)
