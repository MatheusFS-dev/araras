from typing import List, Optional

import os
import numpy as np
import optuna
from tensorflow.keras import callbacks
import tensorflow as tf
from optuna.integration import KerasPruningCallback
from optuna.terminator import BaseImprovementEvaluator, RegretBoundEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS

from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()


class ImprovementStagnation:
    """Stop a study when improvement variance plateaus.

    After each completed trial this callback estimates the potential future
    improvement using an ``optuna.terminator`` evaluator. It monitors the
    variance of the last ``window_size`` improvement values and stops the study
    via :meth:`optuna.Study.stop` once the variance drops below
    ``variance_threshold`` after ``min_n_trials`` trials.

    Args:
        min_n_trials (Any): Minimum number of completed trials before variance checks
            are performed.
        window_size (Any): Number of recent improvement values used to compute the
            variance.
        variance_threshold (Any): Threshold below which the variance of improvements
            indicates stagnation.
        improvement_evaluator (Any): Custom improvement evaluator. Defaults to
            :class:`RegretBoundEvaluator`.
    """

    def __init__(
        self,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        window_size: int = 10,
        variance_threshold: float = 1e-10,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
        verbose: int = 1,
    ) -> None:
        """Create a new callback instance.

        Args:
            min_n_trials (int): Minimum number of completed trials before the
                stagnation checks start.
            window_size (int): Number of recent improvements to use for the variance
                calculation.
            variance_threshold (float): Variance value below which the study will be
                stopped.
            improvement_evaluator (Optional[BaseImprovementEvaluator]): Custom evaluator used to compute expected
                improvement. Defaults to :class:`RegretBoundEvaluator` when
                ``None``.
            verbose (int): Whether to log debugging information at each check.
        """
        if improvement_evaluator is None:
            improvement_evaluator = RegretBoundEvaluator()
        self.min_n_trials = min_n_trials
        self.window_size = window_size
        self._variance_threshold = variance_threshold
        self.improvement_evaluator = improvement_evaluator
        self._completed_trials: List[optuna.trial.FrozenTrial] = []
        self._improvements: List[float] = []
        self.verbose = verbose

        vp.verbose = self.verbose

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
        """Evaluate trial results and stop the study if stagnation is detected.

        Args:
            study (optuna.Study): The Optuna study currently being optimised.
            trial (optuna.trial.FrozenTrial): The trial that has just completed.

        Notes:
            This method mutates internal state each time it is called and will
            invoke :meth:`optuna.study.Study.stop` when the variance of recent
            improvements falls below ``variance_threshold``.
        """
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        self._completed_trials.append(trial)

        improvement = self.improvement_evaluator.evaluate(
            trials=self._completed_trials,
            study_direction=study.direction,
        )
        self._improvements.append(improvement)

        if len(self._completed_trials) < max(self.min_n_trials, self.window_size):
            vp.printf(
                f"Not enough trials completed yet to check for stagnation.",
                tag="[Improvement Stagnation Callback] ",
                color="yellow",
            )
            return

        recent_improvements = self._improvements[-self.window_size :]
        variance = float(np.var(recent_improvements))
        
        vp.printf(
            f"Study {study.study_name} – "
            f"Variance of recent improvements: {variance:.3e}",
            tag="[Improvement Stagnation Callback] ",
            color="yellow",
        )

        if variance <= self.variance_threshold:
            vp.printf(
                f"\nStopping study {study.study_name} due to stagnation "   
                f"(variance={variance:.2e} < threshold={self.variance_threshold:.2e})",
                tag="[Improvement Stagnation Callback] ",
                color="yellow",
            )

            study.stop()


class StopIfKeepBeingPruned:
    """
    A callback for Optuna studies that stops the optimization process
    when a specified number of consecutive trials are pruned.

    Args:
        threshold (int): The number of consecutive pruned trials required to stop the study.
    """

    def __init__(self, threshold: int):
        """
        Initializes the callback with the pruning threshold.

        Args:
            threshold (int): The number of consecutive pruned trials required to stop the study.
        """
        self.threshold = threshold
        self._consequtive_pruned_count = (
            0  # Tracks the count of consecutive pruned trials.
        )

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        """
        Invoked after each trial to check its state and decide whether to stop the study.

        Args:
            study (optuna.study.Study): The Optuna study object.
            trial (optuna.trial.FrozenTrial): The trial object containing the state of the trial.
        """
        # Increment the count if the trial was pruned; reset otherwise.
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        # Stop the study if the threshold of consecutive pruned trials is reached.
        if self._consequtive_pruned_count >= self.threshold:
            study.stop()


class StopWhenNoValueImprovement:
    """Stop a study if the objective value fails to improve for ``patience`` trials.

    The callback compares the study's best objective value after each completed
    trial with the best value seen so far. If no improvement exceeding
    ``min_delta`` occurs for ``patience`` consecutive completed trials, the
    optimization is halted via :meth:`optuna.study.Study.stop`.

    Notes:
        Only trials that finish successfully are considered when counting the
        patience window. Pruned or failed trials reset neither the patience
        counter nor the best value.

    Args:
        patience (Any): Number of consecutive completed trials allowed without a new
            best value.
        min_delta (Any): Minimum absolute improvement required to reset the patience
            counter. Defaults to ``0.0``.
        verbose (int): Whether to log a message when stopping the study.

    Raises:
        ValueError: If ``patience`` is not positive or ``min_delta`` is negative.
    """

    def __init__(
        self, patience: int, min_delta: float = 0.0, verbose: int = 0
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self._best_value: Optional[float] = None
        self._counter: int = 0

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        """Update best value and stop the study if no improvement is observed."""

        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if self._best_value is None:
            self._best_value = trial.value
            self._counter = 0
            return

        direction = study.direction
        if direction == optuna.study.StudyDirection.MINIMIZE:
            improvement = self._best_value - trial.value
        else:
            improvement = trial.value - self._best_value

        if improvement > self.min_delta:
            self._best_value = trial.value
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            vp.printf(
                f"Stopping study {study.study_name} after "
                f"{self.patience} trials without improvement",
                tag="[No Value Improvement Callback] ",
                color="yellow",
            )
            study.stop()


class NanLossPrunerOptuna(callbacks.Callback):
    """Prune an Optuna trial when the training loss diverges to ``NaN``.

    Args:
        trial (Any): Optuna trial instance associated with the current Keras model
            run.

    Examples:
        >>> model.fit(..., callbacks=[NanLossPrunerOptuna(trial)])
    """

    def __init__(self, trial: optuna.Trial) -> None:
        """
        Initializes the callback with the Optuna trial reference.

        Args:
            trial (optuna.Trial): The trial object to report and potentially prune.
        """
        super().__init__()  # Initialize the base Keras Callback
        self.trial = trial  # Save trial reference for reporting/pruning

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Called automatically at the end of each training epoch.

        If training loss is NaN, the trial is reported and pruned.

        Args:
            epoch (int): Index of the current epoch.
            logs (dict, optional): Metric results from the epoch (e.g., {"loss": ..., "val_loss": ...}).
        """
        logs = logs or {}  # Use empty dict if `logs` is None
        loss = logs.get("loss")  # Retrieve training loss from logs

        # If loss is NaN, report and prune the trial
        if loss is not None and np.isnan(loss):
            self.trial.report(loss, step=epoch)  # Inform Optuna of the metric value
            raise optuna.exceptions.TrialPruned("Trial pruned due to NaN loss.")


def get_callbacks_study(
    trial: optuna.Trial,
    tensorboard_logs: str | None = None,
    monitor: str = "val_loss",
    early_stopping_patience: int | None = 5,
    reduce_lr_patience: int | None = 3,
    pruning_interval: int | None = 5,
    verbose: int = 0,
) -> List[tf.keras.callbacks.Callback]:
    """Return Optuna-specific training callbacks.

    This helper generates a list of callbacks tailored for Optuna trials. In
    addition to common Keras callbacks, it adds pruning utilities that work with
    :mod:`optuna`. ``EarlyStopping`` and ``ReduceLROnPlateau`` are optional and
    can be disabled by passing ``None`` for their patience values. The interval
    at which the :class:`KerasPruningCallback` evaluates the monitored metric can
    also be customised or disabled entirely.

    Warning:
        TensorBoard callback can cause high memory usage. The ``write_graph``
        option is set to ``False`` because enabling it drastically increases
        memory consumption.

    Args:
        trial (optuna.Trial): The current Optuna trial object.
        tensorboard_logs (str | None): Directory where TensorBoard logs will be stored. If
            ``None``, the TensorBoard callback is omitted.
        monitor (str): The metric to monitor for early stopping and learning rate
            reduction.
        early_stopping_patience (int | None): Number of epochs with no improvement after
            which training will be stopped. Set to ``None`` to disable the
            ``EarlyStopping`` callback. Defaults to ``5``.
        reduce_lr_patience (int | None): Number of epochs with no improvement before the
            learning rate is reduced. Set to ``None`` to disable the
            ``ReduceLROnPlateau`` callback. Defaults to ``2``.
        pruning_interval (int | None): Frequency (in epochs) at which the
            ``KerasPruningCallback`` checks the monitored metric. Set to ``None``
            to disable pruning. Defaults to ``3``.
        verbose (int): Verbosity level for logging within callbacks.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of callbacks to pass into ``model.fit``.

    Raises:
        None.
    """

    callbacks_list: List[tf.keras.callbacks.Callback] = []

    if early_stopping_patience is not None:
        early_stopping = callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=verbose,
        )
        callbacks_list.append(early_stopping)

    if reduce_lr_patience is not None:
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=reduce_lr_patience,
            factor=0.2,
            min_lr=1e-10,
            verbose=verbose,
        )
        callbacks_list.append(reduce_lr)

    #! ——————— WARNING: the callbacks below do not work with multi-objective —————— !#
    # Custom callback to prune trial if NaN loss is encountered
    # nan_pruner_callback = callbacks.TerminateOnNaN()
    nan_loss_pruner_callback = NanLossPrunerOptuna(trial=trial)

    # Optuna's built-in pruning callback for early trial termination
    pruning_callback: Optional[callbacks.Callback] = None
    if pruning_interval is not None:
        pruning_callback = KerasPruningCallback(
            trial, monitor, interval=pruning_interval
        )
    #! ———————————————————————————————————————————————————————————————————————————— !#

    callbacks_list.append(nan_loss_pruner_callback)
    if pruning_callback is not None:
        callbacks_list.append(pruning_callback)

    if tensorboard_logs is not None:
        trial_log_dir = os.path.join(tensorboard_logs, f"trial_{trial.number}")
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=trial_log_dir,
            histogram_freq=1,
            write_graph=False,
            write_images=True,
            update_freq="epoch",
        )
        callbacks_list.append(tensorboard_cb)

    return callbacks_list
