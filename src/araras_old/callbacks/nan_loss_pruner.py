"""
This module provides a custom Keras callback for pruning an Optuna trial when the training loss becomes NaN.
It is particularly useful for early stopping of trials with unstable or diverging training.

Usage:
    Add NanLossPrunerCallback(trial) to your Keras callbacks list in model.fit().
"""

import numpy as np
import optuna
from tensorflow.keras import callbacks


class NanLossPrunerCallback(callbacks.Callback):
    """
    A custom Keras callback that prunes an Optuna trial if NaN is encountered in training loss.

    This is useful for skipping unpromising model configurations early, especially
    those that are unstable or diverging during training.

    Args:
        trial (optuna.Trial): The Optuna trial associated with this model run.

    Example:
        model.fit(..., callbacks=[NanLossPrunerCallback(trial)])
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
