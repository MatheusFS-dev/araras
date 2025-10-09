from typing import List

import os

from tensorflow.keras import callbacks
import tensorflow as tf


def get_callbacks_model(
    backup_dir: str | None = None,
    checkpoint_dir: str | None = None,
    tensorboard_logs: str | None = None,
    early_stopping_patience: int | None = 10,
    reduce_lr_patience: int | None = 5,
    restore_best_weights: bool = True,
    verbose: int = 1,
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
        backup_dir (str | None): Directory where the backup files will be stored. If
            ``None``, the backup callback is omitted.
        checkpoint_dir (str | None): Directory where the checkpoint files will be stored. If
            ``None``, the checkpoint callback is omitted.
        tensorboard_logs (str | None): Directory where TensorBoard logs will be stored. If
            ``None``, the TensorBoard callback is omitted.
        early_stopping_patience (int | None): Number of epochs with no improvement after
            which training will be stopped. Set to ``None`` to disable the
            ``EarlyStopping`` callback. Defaults to ``10``.
        reduce_lr_patience (int | None): Number of epochs with no improvement before the
            learning rate is reduced. Set to ``None`` to disable the
            ``ReduceLROnPlateau`` callback. Defaults to ``5``.
        restore_best_weights (bool): Whether to restore model weights from the epoch
            with the best monitored metric. When ``True`` and
            ``early_stopping_patience`` is ``None``, ``checkpoint_dir`` **must**
            be provided so the best weights can be reloaded at the end of
            training. Defaults to ``True``.
        verbose (int): Verbosity mode for keras callbacks.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of callbacks to pass into ``model.fit``.

    Raises:
        ValueError: If ``restore_best_weights`` is ``True`` while both
            ``early_stopping_patience`` and ``checkpoint_dir`` are ``None``.
    """
    # Metric to monitor for early stopping and checkpointing
    monitor: str = "val_loss"

    callbacks_list: List[tf.keras.callbacks.Callback] = []

    if restore_best_weights and early_stopping_patience is None and checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir must be provided when restore_best_weights is True and early_stopping_patience is None"
        )

    if early_stopping_patience is not None:
        early_stopping = callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
        )
        callbacks_list.append(early_stopping)
    elif restore_best_weights:
        class RestoreBestWeights(callbacks.Callback):
            def on_train_end(self, logs=None):
                self.model.load_weights(os.path.join(checkpoint_dir, ".weights.h5"))

        callbacks_list.append(RestoreBestWeights())

    if reduce_lr_patience is not None:
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=reduce_lr_patience,
            factor=0.2,
            min_lr=1e-8,
            verbose=verbose,
        )
        callbacks_list.append(reduce_lr)

    if checkpoint_dir is not None:
        checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, ".weights.h5"),
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

    if backup_dir is not None:
        backup_cb = callbacks.BackupAndRestore(
            backup_dir=backup_dir,
            save_freq="epoch",
            delete_checkpoint=True,
        )
        callbacks_list.append(backup_cb)

    return callbacks_list
