from typing import Callable, Sequence
from araras.core import *

import itertools
import optuna
import tensorflow as tf
from tensorflow.keras import layers


def _unique_name(base: str) -> str:
    """Create a globally unique Keras layer name.

    Notes:
        Utilizes :func:`tf.keras.backend.get_uid` to append an index that
        guarantees uniqueness, even when the same ``base`` is provided across
        multiple calls or models.

    Args:
        base: Suggested base string for the layer name.

    Returns:
        A unique layer name derived from ``base``.

    Raises:
        ValueError: If ``base`` is empty.
    """

    if not base:
        raise ValueError("base must be a non-empty string.")
    return f"{base}_{tf.keras.backend.get_uid(base)}"


def _resize_1d(x: tf.Tensor, length: int, name: str) -> tf.Tensor:
    """Resize a 1D tensor to a specific length using nearest interpolation.

    Args:
        x: Input tensor of shape ``(batch, length, channels)``.
        length: Target temporal length.
        name: Base name for the generated :class:`~keras.layers.Lambda` layer.

    Returns:
        Tensor resized along the temporal axis.
    """

    def _func(t: tf.Tensor) -> tf.Tensor:
        t = tf.expand_dims(t, axis=2)
        t = tf.image.resize(t, (length, 1), method="nearest")
        return tf.squeeze(t, axis=2)

    return layers.Lambda(_func, name=name)(x)


def _project_conv1d(
    source: tf.Tensor,
    target: tf.Tensor,
    use_batch_norm: bool,
    name: str,
) -> tf.Tensor:
    """Project a source 1-D tensor so it matches a target tensor.

    The function applies a ``Conv1D`` with kernel size ``1`` to adjust the
    channel dimension and optionally downsamples the temporal axis. When the
    output length still differs from ``target`` a resize operation is applied.

    Args:
        source: Tensor with shape ``(batch, length, channels)``.
        target: Tensor whose shape provides the desired length and channels.
        use_batch_norm: Whether to include a batch-normalization layer after the
            convolution.
        name: Base name used for created layers.

    Returns:
        The projected tensor that has the same shape as ``target``.
    """

    len_tgt = target.shape[1]
    channels_tgt = target.shape[2]
    stride = 1
    len_src = source.shape[1]
    if (
        len_src is not None
        and len_tgt is not None
        and len_src >= len_tgt
        and len_src % len_tgt == 0
    ):
        stride = len_src // len_tgt

    x = layers.Conv1D(
        channels_tgt,
        kernel_size=1,
        strides=stride,
        padding="same",
        name=f"{name}_conv",
    )(source)

    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)

    if len_tgt is not None and x.shape[1] is not None and x.shape[1] != len_tgt:
        x = _resize_1d(x, len_tgt, name=f"{name}_resize")

    return x


def _project_dense(
    source: tf.Tensor,
    target: tf.Tensor,
    use_batch_norm: bool,
    name: str,
) -> tf.Tensor:
    """Project a dense tensor to match another tensor's feature dimension.

    Args:
        source: Input tensor with shape ``(batch, features)``.
        target: Tensor whose last dimension defines the desired ``features``.
        use_batch_norm: If ``True``, apply batch normalization after the dense
            projection.
        name: Base name used for created layers.

    Returns:
        Tensor with the same feature dimension as ``target``.
    """

    units = target.shape[-1]
    x = layers.Dense(units, name=f"{name}_dense")(source)

    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)

    return x

def _trial_skip_connections_projected(
    trial: optuna.trial.Trial,
    layers_list: Sequence[tf.Tensor],
    project: Callable[[tf.Tensor, tf.Tensor, str], tf.Tensor],
    *,
    axis_to_concat: int = -1,
    print_combinations: bool = False,
    strategy: str = "final",
    merge_mode: str = "add",
    name_prefix: str = "skip_proj",
) -> tf.Tensor:
    """Generic skip connections that project mismatched tensors.

    Layer names are automatically suffixed with unique identifiers so the
    function can be invoked multiple times in the same model without causing
    naming collisions.

    Args:
        trial: Optuna trial for selecting which skips are active.
        layers_list: Sequence of tensors to connect.
        project: Callable used to adapt a source tensor to a target tensor.
        axis_to_concat: Axis for concatenation when ``merge_mode`` is ``"concat"``.
        print_combinations: Whether to print every skip configuration.
        strategy: ``"final"`` to only skip to the final layer or ``"any"`` for all
            forward pairs.
        merge_mode: ``"add"`` or ``"concat"``.
        name_prefix: Prefix for layers created by ``project``.

    Returns:
        Tensor after applying the selected skip connections.

    Raises:
        ValueError: If ``strategy`` or ``merge_mode`` are invalid.
    """

    N = len(layers_list)
    if N < 2:
        return layers_list[-1]

    last_idx = N - 1

    if strategy == "any":
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    elif strategy == "final":
        pairs = [(i, last_idx) for i in range(last_idx)]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'final' or 'any'.")

    num_skips = len(pairs)
    if print_combinations:
        print("=" * 50)
        print(f"Total skip possibilities: {2**num_skips}")
        for combo in itertools.product([False, True], repeat=num_skips):
            settings = {f"skip_{i}_{j}": val for (i, j), val in zip(pairs, combo)}
            print(settings)
        print("=" * 50)

    if merge_mode not in ("concat", "add"):
        raise ValueError(f"Unknown merge_mode '{merge_mode}'. Use 'concat' or 'add'.")

    if strategy == "final":
        selected = []
        for i in range(last_idx):
            include = trial.suggest_categorical(f"skip_{i}_{last_idx}", [False, True])
            if include:
                src = project(
                    layers_list[i],
                    layers_list[-1],
                    _unique_name(f"{name_prefix}_{i}_{last_idx}"),
                )
                selected.append(src)
        if not selected:
            return layers_list[-1]
        selected.append(layers_list[-1])
        if merge_mode == "concat":
            layer_name = _unique_name(f"{name_prefix}_concat_final")
            return layers.Concatenate(axis=axis_to_concat, name=layer_name)(selected)
        layer_name = _unique_name(f"{name_prefix}_add_final")
        return layers.Add(name=layer_name)(selected)

    updated = list(layers_list)
    for j in range(1, N):
        sources = []
        for i in range(j):
            include = trial.suggest_categorical(f"skip_{i}_{j}", [False, True])
            if include:
                src = project(
                    updated[i],
                    updated[j],
                    _unique_name(f"{name_prefix}_{i}_{j}"),
                )
                sources.append(src)
        if not sources:
            continue
        sources.append(updated[j])
        if merge_mode == "concat":
            layer_name = _unique_name(f"{name_prefix}_concat_{j}")
            updated[j] = layers.Concatenate(axis=axis_to_concat, name=layer_name)(sources)
        else:
            layer_name = _unique_name(f"{name_prefix}_add_{j}")
            updated[j] = layers.Add(name=layer_name)(sources)

    return updated[-1]


def trial_skip_connections_cnn1d(
    trial: optuna.trial.Trial,
    layers_list: Sequence[tf.Tensor],
    axis_to_concat: int = -1,
    use_batch_norm: bool = False,
    print_combinations: bool = False,
    strategy: str = "final",
    merge_mode: str = "add",
    name_prefix: str = "skip_proj",
) -> tf.Tensor:
    """Apply projected skip connections for 1D convolutional tensors.

    Notes:
        The projection uses ``Conv1D(1)`` layers to match channel dimensions and
        optionally downsample the temporal axis when the source length is an
        integer multiple of the target length.

    Args:
        trial: Optuna trial for selecting which skips to include.
        layers_list: Sequence of tensors to connect via skips. Each tensor must
            have rank 3 ``(batch, length, channels)``.
        axis_to_concat: Concatenation axis if ``merge_mode`` is ``"concat"``.
        use_batch_norm: Whether to apply batch normalization in the projection
            branch.
        print_combinations: Display all possible skip configurations when ``True``.
        strategy: ``"final"`` to only skip to the last layer or ``"any"`` for all
            forward pairs.
        merge_mode: ``"add"`` or ``"concat"``.
        name_prefix: Prefix for projection layer names.

    Returns:
        The merged output tensor after applying the selected skip connections.

    Raises:
        ValueError: If ``strategy`` or ``merge_mode`` contain invalid values.
    """

    return _trial_skip_connections_projected(
        trial=trial,
        layers_list=layers_list,
        project=lambda s, t, name: _project_conv1d(s, t, use_batch_norm, name),
        axis_to_concat=axis_to_concat,
        print_combinations=print_combinations,
        strategy=strategy,
        merge_mode=merge_mode,
        name_prefix=name_prefix,
    )


def trial_skip_connections_dense(
    trial: optuna.trial.Trial,
    layers_list: Sequence[tf.Tensor],
    axis_to_concat: int = -1,
    use_batch_norm: bool = False,
    print_combinations: bool = False,
    strategy: str = "final",
    merge_mode: str = "add",
    name_prefix: str = "skip_proj",
    ) -> tf.Tensor:
    """Skip connections for dense tensors with automatic feature projection.

    Args:
        trial: Optuna trial governing which connections are active.
        layers_list: Sequence of 2D tensors ``(batch, features)``.
        axis_to_concat: Axis used when concatenating outputs.
        use_batch_norm: Whether to apply batch normalization in projection
            branches.
        print_combinations: If ``True``, print all skip configurations.
        strategy: ``"final"`` or ``"any"``.
        merge_mode: Either ``"add"`` or ``"concat"``.
        name_prefix: Prefix for projection layer names.

    Returns:
        Output tensor after skip connections.

    Raises:
        ValueError: If ``strategy`` or ``merge_mode`` are invalid.
    """

    return _trial_skip_connections_projected(
        trial=trial,
        layers_list=layers_list,
        project=lambda s, t, name: _project_dense(s, t, use_batch_norm, name),
        axis_to_concat=axis_to_concat,
        print_combinations=print_combinations,
        strategy=strategy,
        merge_mode=merge_mode,
        name_prefix=name_prefix,
    )