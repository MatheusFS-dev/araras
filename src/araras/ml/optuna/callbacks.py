"""
Last Edited: 14 July 2025
Description:
    Optuna callbacks to track trial progress.
"""
from araras.core import *

import os
import numpy as np
import optuna
from tensorflow.keras import callbacks
import tensorflow as tf
from optuna.integration import KerasPruningCallback
from optuna.terminator import BaseImprovementEvaluator, RegretBoundEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS


class ImprovementStagnation:
    """Stop a study when improvement variance plateaus.

    After each completed trial this callback estimates the potential future
    improvement using an ``optuna.terminator`` evaluator. It monitors the
    variance of the last ``window_size`` improvement values and stops the study
    via :meth:`optuna.Study.stop` once the variance drops below
    ``variance_threshold`` after ``min_n_trials`` trials.

    Args:
        min_n_trials: Minimum number of completed trials before variance checks
            are performed.
        window_size: Number of recent improvement values used to compute the
            variance.
        variance_threshold: Threshold below which the variance of improvements
            indicates stagnation.
        improvement_evaluator: Custom improvement evaluator. Defaults to
            :class:`RegretBoundEvaluator`.

    Returns:
        None

    Raises:
        None
    """

    def __init__(
        self,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        window_size: int = 5,
        variance_threshold: float = 1e-10,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
        verbose: bool = False,
    ) -> None:
        if improvement_evaluator is None:
            improvement_evaluator = RegretBoundEvaluator()
        self.min_n_trials = min_n_trials
        self.window_size = window_size
        self._variance_threshold = variance_threshold
        self.improvement_evaluator = improvement_evaluator
        self._completed_trials: List[optuna.trial.FrozenTrial] = []
        self._improvements: List[float] = []
        self.verbose = verbose

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
            if self.verbose:
                logger.warning(
                    "[Improvement Stagnation Callback] Not enough trials completed yet to check for stagnation."
                )
            return

        recent_improvements = self._improvements[-self.window_size :]
        variance = float(np.var(recent_improvements))

        if self.verbose:
            logger.warning(
                f"[Improvement Stagnation Callback] Study {study.study_name} – "
                f"Variance of recent improvements: {variance:.3e}"
            )

        if variance <= self.variance_threshold:
            logger.warning(
                f"\033[33m\n[Improvement Stagnation Callback] Stopping study {study.study_name} due to stagnation "
                f"(variance={variance:.2e} < threshold={self.variance_threshold:.2e})\033[0m\n"
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
        self._consequtive_pruned_count = 0  # Tracks the count of consecutive pruned trials.

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
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


class NanLossPrunerOptuna(callbacks.Callback):
    """
    A custom Keras callback that prunes an Optuna trial if NaN is encountered in training loss.

    This is useful for skipping unpromising model configurations early, especially
    those that are unstable or diverging during training.

    Args:
        trial (optuna.Trial): The Optuna trial associated with this model run.

    Example:
        model.fit(..., callbacks=[NanLossPrunerOptuna(trial)])
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
    trial: optuna.Trial, tensorboard_logs: str = None, monitor: str = "val_loss"
) -> List[tf.keras.callbacks.Callback]:
    """
    Constructs and returns a list of Keras callbacks tailored for Optuna trials.

    Args:
        trial (optuna.Trial): The current Optuna trial object.
        tensorboard_logs (str): Directory where TensorBoard logs will be stored.
        monitor (str): The metric to monitor for early stopping and learning rate reduction.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.
    """

    # Stop training early if no improvement in validation loss for N epochs
    early_stopping = callbacks.EarlyStopping(
        monitor=monitor,
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    # Reduce learning rate if validation loss plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor=monitor,
        patience=2,
        factor=0.2,  # reduce LR by this factor
        min_lr=1e-10,  # don't reduce below this
        verbose=0,
    )

    if tensorboard_logs is not None:
        trial_log_dir = os.path.join(tensorboard_logs, f"trial_{trial.number}")
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=trial_log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch"
        )

    #! ——————— WARNING: the callbacks below do not work with multi-objective —————— !#
    # Custom callback to prune trial if NaN loss is encountered
    # nan_pruner_callback = callbacks.TerminateOnNaN()
    nan_loss_pruner_callback = NanLossPrunerOptuna(trial=trial)

    # Optuna's built-in pruning callback for early trial termination
    pruning_callback = KerasPruningCallback(trial, monitor, interval=3)
    #! ———————————————————————————————————————————————————————————————————————————— !#

    if tensorboard_logs is not None:
        return [early_stopping, reduce_lr, nan_loss_pruner_callback, pruning_callback, tensorboard_cb]
    else:
        return [early_stopping, reduce_lr, nan_loss_pruner_callback, pruning_callback]
