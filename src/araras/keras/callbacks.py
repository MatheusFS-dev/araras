"""
Keras callback for pruning Optuna trials when the training loss becomes NaN.
It does the same as `keras.callbacks.TerminateOnNaN()`, but also reports the NaN loss to Optuna.

Classes:
    - NanLossPrunerOptuna: Stops a trial once `loss` is NaN.

Example:
    >>> from araras.keras.callbacks.nan_loss_pruner import NanLossPrunerOptuna
    >>> NanLossPrunerOptuna(trial)
"""

from araras.commons import *  # Common imports and configs for the Araras lib

import numpy as np
import optuna
from tensorflow.keras import callbacks


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
