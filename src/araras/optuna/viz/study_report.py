from __future__ import annotations

"""Realtime study reporting utilities using Plotly."""

from typing import Dict, List, Optional, Any

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

        trial_numbers.append(len(trial_numbers) + 1)

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
    """Utility class for realtime Optuna study reporting.

    The class stores metric histories and displays Plotly graphs in a
    standalone browser window. Each call to :meth:`update` appends new values
    and re-renders the figure, allowing realtime feedback while Optuna trials
    run.
    """

    def __init__(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        summary_values: Optional[Dict[str, Any]] = None,
        best_is_min: bool = True,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
        error_evaluator: Optional[BaseErrorEvaluator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self.metric_history: Dict[str, List[Any]] = {}
        self.summary_history: Dict[str, List[Any]] = {}
        if metrics:
            for name, val in metrics.items():
                self.metric_history[name] = [val]
        if summary_values:
            for name, val in summary_values.items():
                self.summary_history[name] = [val]
        self.best_is_min = best_is_min
        self.improvement_evaluator = improvement_evaluator
        self.error_evaluator = error_evaluator
        self.min_n_trials = min_n_trials

        self.best_metrics: Dict[str, float] = {}

        self._fig: Optional[go.Figure] = None

    # ------------------------------------------------------------------
    def update(
        self,
        study: optuna.Study,
        metrics: Optional[Dict[str, Any]] = None,
        summary_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update and display graphs and summary for the given study."""

        metrics = metrics or {}
        summary_values = summary_values or {}

        # ------------------------------------------------------------------
        # Store metric and summary values
        for name, val in metrics.items():
            self.metric_history.setdefault(name, []).append(val)
            best = self.best_metrics.get(name)
            if best is None or (self.best_is_min and val < best) or (not self.best_is_min and val > best):
                self.best_metrics[name] = val

        for name, val in summary_values.items():
            self.summary_history.setdefault(name, []).append(val)

        # ------------------------------------------------------------------
        # Improvement and error curves
        trial_nums, improvements, errors = _get_improvement_and_error(
            study,
            improvement_evaluator=self.improvement_evaluator,
            error_evaluator=self.error_evaluator,
        )

        if improvements:
            best_impr = min(improvements) if self.best_is_min else max(improvements)
            prev = self.best_metrics.get("improvement")
            if (
                prev is None
                or (self.best_is_min and best_impr < prev)
                or (not self.best_is_min and best_impr > prev)
            ):
                self.best_metrics["improvement"] = best_impr

        # ------------------------------------------------------------------
        # Build figure
        rows = 1 + len(self.metric_history)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)

        x_impr = list(range(1, len(improvements) + 1))
        if x_impr:
            fig.add_trace(
                go.Scatter(x=x_impr, y=improvements, mode="lines+markers", name="Improvement"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x_impr, y=errors, mode="lines+markers", name="Error"),
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="Improvement", row=1, col=1, autorange=True)

        row_idx = 2
        for name, values in self.metric_history.items():
            x_vals = list(range(1, len(values) + 1))
            fig.add_trace(
                go.Scatter(x=x_vals, y=values, mode="lines+markers", name=name),
                row=row_idx,
                col=1,
            )
            fig.update_yaxes(title_text=name, row=row_idx, col=1, autorange=True)
            row_idx += 1

        fig.update_xaxes(title_text="Trial", row=rows, col=1)
        max_len = max([len(improvements)] + [len(v) for v in self.metric_history.values()] + [1])
        fig.update_xaxes(range=[1, max_len], row=rows, col=1)
        fig.update_layout(height=300 * rows, showlegend=True)

        fig.show(renderer="browser")

        # ------------------------------------------------------------------
        # Print summary
        summary_lines: List[str] = []
        for name, hist in self.summary_history.items():
            summary_lines.append(f"{name}: {hist[-1]}")

        for name, best in self.best_metrics.items():
            summary_lines.append(f"Best {name}: {best}")

        print("\n".join(summary_lines))


_report_instance: Optional[StudyReport] = None


def report(
    study: optuna.Study,
    metrics: Optional[Dict[str, Any]] = None,
    summary_values: Optional[Dict[str, Any]] = None,
    best_is_min: bool = True,
    improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
    error_evaluator: Optional[BaseErrorEvaluator] = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> StudyReport:
    """Update or create a :class:`StudyReport` for ``study``.

    Parameters
    ----------
    study:
        The :class:`optuna.Study` being optimized.
    metrics:
        Mapping of metric labels to numeric values for the current trial.
        Each value is appended to a history and plotted over time.
    summary_values:
        Mapping of labels to numeric values summarizing the study. These are
        displayed below the graphs.
    best_is_min:
        If ``True`` lower values are considered better when tracking the best
        metrics.
    improvement_evaluator, error_evaluator:
        Optional custom evaluators used to compute the improvement/error curves.
    min_n_trials:
        Minimum number of trials before the improvement calculation kicks in.

    Examples
    --------
    >>> study = optuna.create_study(direction="minimize")
    >>> def objective(trial):
    ...     x = trial.suggest_float("x", -5, 5)
    ...     loss = x ** 2
    ...     report(
    ...         study,
    ...         metrics={"loss": loss},
    ...         summary_values={"Completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])},
    ...     )
    ...     return loss
    >>> study.optimize(objective, n_trials=10)
    """

    global _report_instance

    if _report_instance is None:
        _report_instance = StudyReport(
            best_is_min=best_is_min,
            improvement_evaluator=improvement_evaluator,
            error_evaluator=error_evaluator,
            min_n_trials=min_n_trials,
        )

    _report_instance.update(study, metrics=metrics, summary_values=summary_values)
    return _report_instance
