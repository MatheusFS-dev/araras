"""
Keras callback for pruning Optuna trials when the training loss becomes NaN.
It does the same as `keras.callbacks.TerminateOnNaN()`, but also reports the NaN loss to Optuna.

Classes:
    - NanLossPrunerOptuna: Stops a trial once `loss` is NaN.
    
Functions:
    - get_callbacks_study: Returns a list of Keras callbacks for Optuna trials.
    - get_callbacks_model: Returns a list of Keras callbacks for model training.

Example:
    >>> from araras.keras.callbacks.nan_loss_pruner import NanLossPrunerOptuna
    >>> NanLossPrunerOptuna(trial)
"""

from araras.commons import *  # Common imports and configs for the Araras lib

import os
import numpy as np
import optuna
from tensorflow.keras import callbacks
import tensorflow as tf
from optuna.integration import KerasPruningCallback


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


def get_callbacks_model(backup_dir: str, tensorboard_logs: str) -> List[tf.keras.callbacks.Callback]:
    """
    Constructs and returns a list of Keras callbacks for model training.

    Args:
        backup_dir (str): Directory where the backup files will be stored.
        tensorboard_logs (str): Directory where TensorBoard logs will be stored.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of callbacks to pass into `model.fit()`.
    """
    # Metric to monitor for early stopping and checkpointing
    monitor: str = "val_loss"

    # Stop training early if no improvement in validation loss for N epochs
    early_stopping = callbacks.EarlyStopping(
        monitor=monitor,
        patience=10,  # number of epochs to wait
        restore_best_weights=True,
        verbose=1,
    )

    # Reduce learning rate if validation loss plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor=monitor,
        patience=5,  # how many epochs to wait before reducing LR
        factor=0.2,  # reduce LR by this factor
        min_lr=1e-8,  # don't reduce below this
        verbose=1,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(backup_dir, "checkpoint.h5"),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
    )

    trial_log_dir = os.path.join(tensorboard_logs, f"tensorboard_logs")
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=trial_log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch"
    )

    return [early_stopping, reduce_lr, checkpoint, tensorboard_cb]
