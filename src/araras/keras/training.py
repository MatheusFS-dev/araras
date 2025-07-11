"""
Collection of functions that help training Keras models.
"""
from araras.commons import *

import os
import optuna
import tensorflow as tf
from optuna.integration import KerasPruningCallback
from tensorflow.keras import callbacks

from araras.keras.callbacks import NanLossPrunerOptuna


def get_callbacks_study(trial: optuna.Trial, tensorboard_logs: str = None, monitor: str = "val_loss") -> List[tf.keras.callbacks.Callback]:
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
            log_dir=trial_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch"
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
        log_dir=trial_log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch"
    )

    return [early_stopping, reduce_lr, checkpoint, tensorboard_cb]
