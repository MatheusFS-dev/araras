"""Callback utilities for Optuna studies."""

from __future__ import annotations

from typing import List, Optional

import optuna
from optuna.terminator import BaseImprovementEvaluator, RegretBoundEvaluator
from optuna.terminator.improvement.evaluator import BestValueStagnationEvaluator, DEFAULT_MIN_N_TRIALS


class ImprovementStagnationCallback:
    """Stop a study when the terminator improvement stagnates.

    This callback evaluates the potential improvement of the study after each
    completed trial using an ``optuna.terminator`` improvement evaluator. If the
    estimated improvement drops to or below ``threshold`` after ``min_n_trials``
    completed trials, the optimization is terminated via :meth:`optuna.Study.stop`.

    Parameters
    ----------
    min_n_trials:
        Minimum number of completed trials before evaluating stagnation.
    threshold:
        Improvement threshold used to determine stagnation. The study is stopped
        when the evaluator returns a value less than or equal to this threshold.
    improvement_evaluator:
        Custom improvement evaluator. Defaults to :class:`RegretBoundEvaluator`.
    """

    def __init__(
        self,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        threshold: float = 0.0,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    ) -> None:
        if improvement_evaluator is None:
            improvement_evaluator = RegretBoundEvaluator()
        self.min_n_trials = min_n_trials
        self.threshold = threshold
        self.improvement_evaluator = improvement_evaluator
        self._completed_trials: List[optuna.trial.FrozenTrial] = []

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        self._completed_trials.append(trial)
        if len(self._completed_trials) < self.min_n_trials:
            return

        improvement = self.improvement_evaluator.evaluate(
            trials=self._completed_trials,
            study_direction=study.direction,
        )

        if improvement <= self.threshold:
            study.stop()
