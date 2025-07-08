"""Callback utilities for Optuna studies."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import optuna
from optuna.terminator import BaseImprovementEvaluator, RegretBoundEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ImprovementStagnationCallback:
    """Stop a study when the terminator improvement variance plateaus.

    After each completed trial the callback computes the potential future
    improvement using an ``optuna.terminator`` improvement evaluator. The
    variance of the most recent ``window_size`` improvement values is measured
    and if it falls below ``variance_threshold`` after ``min_n_trials`` trials
    the study is terminated via :meth:`optuna.Study.stop`.

    Parameters
    ----------
    min_n_trials:
        Minimum number of completed trials before starting variance checks.
    window_size:
        Number of recent improvement values used to compute the variance.
    variance_threshold:
        Threshold below which the variance of improvements indicates stagnation.
    improvement_evaluator:
        Custom improvement evaluator. Defaults to :class:`RegretBoundEvaluator`.
    """

    def __init__(
        self,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        window_size: int = 5,
        variance_threshold: float = 1e-10,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    ) -> None:
        if improvement_evaluator is None:
            improvement_evaluator = RegretBoundEvaluator()
        self.min_n_trials = min_n_trials
        self.window_size = window_size
        self._variance_threshold = variance_threshold
        self.improvement_evaluator = improvement_evaluator
        self._completed_trials: List[optuna.trial.FrozenTrial] = []
        self._improvements: List[float] = []

    @property
    def variance_threshold(self) -> float:
        """Variance threshold triggering study stop."""
        return self._variance_threshold

    @variance_threshold.setter
    def variance_threshold(self, value: float) -> None:
        if value < 0:
            raise ValueError("variance_threshold must be non-negative")
        self._variance_threshold = value

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        self._completed_trials.append(trial)

        improvement = self.improvement_evaluator.evaluate(
            trials=self._completed_trials,
            study_direction=study.direction,
        )
        self._improvements.append(improvement)

        if len(self._completed_trials) < max(self.min_n_trials, self.window_size):
            logger.warning("Not enough trials completed yet to check for stagnation.")
            return

        recent_improvements = self._improvements[-self.window_size :]
        variance = float(np.var(recent_improvements))
        
        logger.info(f"Study {study.study_name} – ... Variance: {variance:.3e}")

        if variance <= self.variance_threshold:
            print(
                f"\033[33m\nStopping study {study.study_name} due to stagnation "
                f"(variance={variance:.2e} < threshold={self.variance_threshold:.2e})\033[0m\n"
            )
            
            study.stop()
