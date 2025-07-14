from araras.core import *

import os

from tensorflow.keras import callbacks
import tensorflow as tf


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
