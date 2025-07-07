from __future__ import annotations

"""Realtime study reporting utilities using Plotly."""

from typing import Callable, Dict, List, Optional, Any

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_improvement_and_error(
    study: optuna.Study,
    improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    error_evaluator: Optional[BaseErrorEvaluator] = None,
) -> tuple[List[int], List[float], List[float]]:
    """Return trial numbers, improvement values and error estimates."""

    if study._is_multi_objective():
        raise ValueError("Multi-objective studies are not supported")

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

        if not completed_trials:
            continue

        trial_numbers.append(trial.number)

        improvement = improvement_evaluator.evaluate(
            trials=completed_trials,
            study_direction=study.direction,
        )
        improvements.append(improvement)

        error = error_evaluator.evaluate(
            trials=completed_trials,
            study_direction=study.direction,
        )
        errors.append(error)

    return trial_numbers, improvements, errors


class StudyReport:
    """Utility class for realtime Optuna study reporting."""

    def __init__(
        self,
        metrics: Optional[Dict[str, Callable[[optuna.trial.FrozenTrial], Any]]] = None,
        summary_values: Optional[Dict[str, Callable[[optuna.Study], Any]]] = None,
        best_is_min: bool = True,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
        error_evaluator: Optional[BaseErrorEvaluator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self.metrics = metrics or {}
        self.summary_values = summary_values or {}
        self.best_is_min = best_is_min
        self.improvement_evaluator = improvement_evaluator
        self.error_evaluator = error_evaluator
        self.min_n_trials = min_n_trials

        self.best_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def update(self, study: optuna.Study) -> None:
        """Update and display graphs and summary for the given study."""

        trial_nums, improvements, errors = _get_improvement_and_error(
            study,
            improvement_evaluator=self.improvement_evaluator,
            error_evaluator=self.error_evaluator,
        )

        metric_data: Dict[str, List[Any]] = {name: [] for name in self.metrics}
        metric_trials: List[int] = [
            t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            for name, fn in self.metrics.items():
                try:
                    metric_data[name].append(fn(trial))
                except Exception:
                    metric_data[name].append(None)

        # Update best metrics
        if improvements:
            best_impr = min(improvements) if self.best_is_min else max(improvements)
            prev = self.best_metrics.get("improvement")
            if prev is None or (
                self.best_is_min and best_impr < prev
            ) or (not self.best_is_min and best_impr > prev):
                self.best_metrics["improvement"] = best_impr

        for name, values in metric_data.items():
            clean_vals = [v for v in values if v is not None]
            if not clean_vals:
                continue
            best_val = min(clean_vals) if self.best_is_min else max(clean_vals)
            prev = self.best_metrics.get(name)
            if prev is None or (
                self.best_is_min and best_val < prev
            ) or (not self.best_is_min and best_val > prev):
                self.best_metrics[name] = best_val

        # ------------------------------------------------------------------
        # Build figure
        rows = 1 + len(metric_data)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)

        if trial_nums:
            fig.add_scatter(
                x=trial_nums,
                y=improvements,
                mode="lines+markers",
                name="Improvement",
                row=1,
                col=1,
            )
            fig.add_scatter(
                x=trial_nums,
                y=errors,
                mode="lines+markers",
                name="Error",
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="Improvement", row=1, col=1)

        row_idx = 2
        for name, values in metric_data.items():
            fig.add_scatter(
                x=metric_trials,
                y=values,
                mode="lines+markers",
                name=name,
                row=row_idx,
                col=1,
            )
            fig.update_yaxes(title_text=name, row=row_idx, col=1)
            row_idx += 1

        fig.update_xaxes(title_text="Trial", row=rows, col=1)
        fig.update_layout(height=300 * rows, showlegend=True)
        fig.show()

        # ------------------------------------------------------------------
        # Print summary
        summary_lines: List[str] = []
        for desc, fn in self.summary_values.items():
            try:
                value = fn(study)
            except Exception:
                value = None
            summary_lines.append(f"{desc}: {value}")

        for name, best in self.best_metrics.items():
            summary_lines.append(f"Best {name}: {best}")

        print("\n".join(summary_lines))


_report_instance: Optional[StudyReport] = None


def report(
    study: optuna.Study,
    metrics: Optional[Dict[str, Callable[[optuna.trial.FrozenTrial], Any]]] = None,
    summary_values: Optional[Dict[str, Callable[[optuna.Study], Any]]] = None,
    best_is_min: bool = True,
    improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    error_evaluator: Optional[BaseErrorEvaluator] = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> StudyReport:
    """Update or create a :class:`StudyReport` for the given study."""

    global _report_instance

    if _report_instance is None:
        _report_instance = StudyReport(
            metrics=metrics,
            summary_values=summary_values,
            best_is_min=best_is_min,
            improvement_evaluator=improvement_evaluator,
            error_evaluator=error_evaluator,
            min_n_trials=min_n_trials,
        )

    _report_instance.update(study)
    return _report_instance
