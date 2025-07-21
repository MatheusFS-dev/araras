from araras.core import *

import os

from tensorflow.keras import callbacks
import tensorflow as tf


def get_callbacks_model(
    backup_dir: str,
    tensorboard_logs: str | None = None,
    early_stopping_patience: int | None = 10,
    reduce_lr_patience: int | None = 5,
) -> List[tf.keras.callbacks.Callback]:
    """Return commonly used training callbacks.

    This helper constructs a list of :mod:`keras` callbacks for model training. A
    ``ModelCheckpoint`` callback is always created to persist the best weights.
    ``EarlyStopping`` and ``ReduceLROnPlateau`` callbacks are optional and
    controlled via their respective patience arguments. Setting a patience value
    to ``None`` disables the corresponding callback.

    Warning:
        TensorBoard callback can cause high memory usage. The ``write_graph``
        option in TensorBoard callback is set to ``False`` because enabling it
        drastically increases memory consumption, especially with large models.

    Args:
        backup_dir: Directory where the backup files will be stored.
        tensorboard_logs: Directory where TensorBoard logs will be stored. If
            ``None``, the TensorBoard callback is omitted.
        early_stopping_patience: Number of epochs with no improvement after
            which training will be stopped. Set to ``None`` to disable the
            ``EarlyStopping`` callback. Defaults to ``10``.
        reduce_lr_patience: Number of epochs with no improvement before the
            learning rate is reduced. Set to ``None`` to disable the
            ``ReduceLROnPlateau`` callback. Defaults to ``5``.

    Returns:
        A list of callbacks to pass into ``model.fit``.

    Raises:
        None.
    """
    # Metric to monitor for early stopping and checkpointing
    monitor: str = "val_loss"

    callbacks_list: List[tf.keras.callbacks.Callback] = []

    if early_stopping_patience is not None:
        early_stopping = callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks_list.append(early_stopping)

    if reduce_lr_patience is not None:
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=reduce_lr_patience,
            factor=0.2,
            min_lr=1e-8,
            verbose=1,
        )
        callbacks_list.append(reduce_lr)

    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(backup_dir, ".weights.h5"),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
    )
    callbacks_list.append(checkpoint)

    if tensorboard_logs is not None:
        tensorboard_cb = callbacks.TensorBoard(
            log_dir=tensorboard_logs,
            histogram_freq=1,
            write_graph=False,
            write_images=True,
            update_freq="epoch",
        )
        callbacks_list.append(tensorboard_cb)

    return callbacks_list
