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

    This helper expands the tensor along a spatial axis, applies
    nearest-neighbor resizing, and then removes the added axis.

    Notes:
        A :class:`~keras.layers.Lambda` layer with an explicit ``output_shape``
        is used to avoid deserialization issues.

    Warnings:
        Setting ``length`` to excessively large values may lead to
        considerable memory consumption.

    Args:
        x: Input tensor of shape ``(batch, length, channels)``.
        length: Target temporal length.
        name: Base name for the generated :class:`~keras.layers.Lambda` layer.

    Returns:
        Tensor resized along the temporal axis with shape
        ``(batch, length, channels)``.

    Raises:
        ValueError: If ``length`` is not a positive integer.
    """

    if length <= 0:
        raise ValueError("length must be a positive integer.")

    def _func(t: tf.Tensor) -> tf.Tensor:
        t = tf.expand_dims(t, axis=2)
        t = tf.image.resize(t, (length, 1), method="nearest")
        return tf.squeeze(t, axis=2)

    return layers.Lambda(
        _func,
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=lambda s: (length, s[-1]),
        name=name,
    )(x)


def _resize_2d(x: tf.Tensor, size: tuple[int, int], name: str) -> tf.Tensor:
    """Resize a 2D tensor to a specific spatial size using nearest interpolation.

    The tensor is resized using nearest-neighbor interpolation.

    Notes:
        Employs a :class:`~keras.layers.Lambda` layer with an explicit
        ``output_shape`` to prevent deserialization problems.

    Warnings:
        Very large ``size`` values can cause high memory usage.

    Args:
        x: Input tensor of shape ``(batch, height, width, channels)``.
        size: Tuple ``(height, width)`` indicating the target spatial dimensions.
        name: Base name for the generated :class:`~keras.layers.Lambda` layer.

    Returns:
        Tensor resized along the spatial axes with shape
        ``(batch, height, width, channels)``.

    Raises:
        ValueError: If any dimension in ``size`` is not a positive integer.
    """

    if any(d <= 0 for d in size):
        raise ValueError("size dimensions must be positive integers.")

    def _func(t: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(t, size, method="nearest")

    return layers.Lambda(
        _func,
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=lambda s: (size[0], size[1], s[-1]),
        name=name,
    )(x)


def _needs_projection(
    source: tf.Tensor,
    target: tf.Tensor,
    merge_mode: str,
    axis_to_concat: int,
) -> bool:
    """Determine whether a source tensor must be projected to match a target.

    The decision depends on the merge strategy. For ``"add"`` the entire shape
    must match. For ``"concat"`` only the dimensions other than
    ``axis_to_concat`` are compared.

    Notes:
        The function accepts both tensors and raw shape tuples. Shapes may contain
        ``None`` for unknown dimensions and are normalized via
        :class:`tf.TensorShape` for safe comparison.

    Args:
        source: Tensor or shape tuple proposed as the skip source.
        target: Tensor or shape tuple receiving the skip connection.
        merge_mode: Either ``"add"`` or ``"concat"``.
        axis_to_concat: Axis along which concatenation occurs when
            ``merge_mode`` is ``"concat"``.

    Returns:
        ``True`` if projection is required, ``False`` otherwise.

    Raises:
        ValueError: If ``merge_mode`` is neither ``"add"`` nor ``"concat"``.
    """

    if merge_mode not in {"concat", "add"}:
        raise ValueError("merge_mode must be 'add' or 'concat'.")

    src_shape = tf.TensorShape(getattr(source, "shape", source)).as_list()
    tgt_shape = tf.TensorShape(getattr(target, "shape", target)).as_list()

    if merge_mode == "concat":
        if len(src_shape) != len(tgt_shape):
            return True
        axis = axis_to_concat % len(tgt_shape)
        for idx, (s_dim, t_dim) in enumerate(zip(src_shape, tgt_shape)):
            if idx == axis:
                continue
            if s_dim != t_dim:
                return True
        return False

    return src_shape != tgt_shape


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
    if len_src is not None and len_tgt is not None and len_src >= len_tgt and len_src % len_tgt == 0:
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


def _project_conv2d(
    source: tf.Tensor,
    target: tf.Tensor,
    use_batch_norm: bool,
    name: str,
) -> tf.Tensor:
    """Project a source 2-D tensor so it matches a target tensor.

    The function applies a ``Conv2D`` with kernel size ``1`` to adjust the
    channel dimension and optionally downsamples spatial axes. When the output
    size still differs from ``target``, a resize operation is applied.

    Args:
        source: Tensor with shape ``(batch, height, width, channels)``.
        target: Tensor providing the desired spatial dimensions and channels.
        use_batch_norm: Whether to include batch normalization after the
            convolution.
        name: Base name used for created layers.

    Returns:
        The projected tensor that has the same shape as ``target``.
    """

    h_tgt, w_tgt = target.shape[1], target.shape[2]
    channels_tgt = target.shape[3]
    stride_h = stride_w = 1
    h_src, w_src = source.shape[1], source.shape[2]
    if h_src is not None and h_tgt is not None and h_src >= h_tgt and h_src % h_tgt == 0:
        stride_h = h_src // h_tgt
    if w_src is not None and w_tgt is not None and w_src >= w_tgt and w_src % w_tgt == 0:
        stride_w = w_src // w_tgt

    x = layers.Conv2D(
        channels_tgt,
        kernel_size=1,
        strides=(stride_h, stride_w),
        padding="same",
        name=f"{name}_conv",
    )(source)

    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)

    if (h_tgt is not None and x.shape[1] is not None and x.shape[1] != h_tgt) or (
        w_tgt is not None and x.shape[2] is not None and x.shape[2] != w_tgt
    ):
        x = _resize_2d(x, (h_tgt, w_tgt), name=f"{name}_resize")

    return x


def _project_dense(
    source: tf.Tensor,
    target: tf.Tensor,
    use_batch_norm: bool,
    name: str,
) -> tf.Tensor:
    """Project a tensor along its last axis to match a target's features.

    Notes:
        This function is compatible with tensors of rank ≥2. It applies a
        :class:`~keras.layers.Dense` layer to adjust the feature dimension while
        keeping all other dimensions intact. When ``use_batch_norm`` is ``True``
        a :class:`~keras.layers.BatchNormalization` layer is appended.

    Args:
        source: Input tensor whose last dimension will be projected.
        target: Tensor providing the desired feature dimension.
        use_batch_norm: Apply batch normalization after the dense layer when
            set to ``True``.
        name: Base name used for created layers.

    Returns:
        Tensor with its last dimension matching that of ``target``.
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
    verbose: int = 1,
    strategy: str = "any",
    merge_mode: str = "add",
    name_prefix: str = "skip_proj",
) -> tf.Tensor:
    """Generic skip connections that project mismatched tensors.

    Layer names are automatically suffixed with unique identifiers so the
    function can be invoked multiple times in the same model without causing
    naming collisions. Projection is only applied to source tensors and only
    when their shapes differ from the target (respecting ``axis_to_concat`` in
    concatenation mode).

    Args:
        trial: Optuna trial for selecting which skips are active.
        layers_list: Sequence of tensors to connect.
        project: Callable used to adapt a source tensor to a target tensor.
        axis_to_concat: Axis for concatenation when ``merge_mode`` is ``"concat"``.
        verbose: Verbosity level for skip combinations. ``0`` prints nothing,
            ``1`` prints only the number of combinations and ``2`` prints the
            number plus every combination. Defaults to ``1``.
        strategy: ``"final"`` to only skip to the final layer or ``"any"`` for all
            forward pairs. Defaults to ``"any"``.
        merge_mode: ``"add"`` or ``"concat"``.
        name_prefix: Prefix for layers created by ``project``.

    Returns:
        Tensor after applying the selected skip connections.

    Raises:
        ValueError: If ``verbose`` not in ``{0, 1, 2}`` or if ``strategy`` or
            ``merge_mode`` are invalid.
    """

    if verbose not in {0, 1, 2}:
        raise ValueError("verbose must be 0, 1, or 2.")

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
    if verbose > 0:
        print("=" * 50)
        print(f"Total skip possibilities: {2**num_skips}")
        if verbose > 1:
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
                name = _unique_name(f"{name_prefix}_{i}_{last_idx}")
                src = layers_list[i]
                if _needs_projection(src, layers_list[-1], merge_mode, axis_to_concat):
                    src = project(src, layers_list[-1], name)
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
                name = _unique_name(f"{name_prefix}_{i}_{j}")
                src = updated[i]
                if _needs_projection(src, updated[j], merge_mode, axis_to_concat):
                    src = project(src, updated[j], name)
                sources.append(src)
        if not sources:
            continue
        sources.append(updated[j])
        if merge_mode == "concat":
            layer_name = _unique_name(f"{name_prefix}_concat_{j}")
            
            # Print sources for debugging
            if verbose > 0:
                print(f"Concatenating sources for skip_{j}: {[s.shape for s in sources]}")
            
            updated[j] = layers.Concatenate(axis=axis_to_concat, name=layer_name)(sources)
        else:
            layer_name = _unique_name(f"{name_prefix}_add_{j}")
            updated[j] = layers.Add(name=layer_name)(sources)

    return updated[-1]


def trial_skip_3d_tensors(
    trial: optuna.trial.Trial,
    layers_list: Sequence[tf.Tensor],
    axis_to_concat: int = -1,
    use_batch_norm: bool = False,
    verbose: int = 1,
    strategy: str = "any",
    merge_mode: str = "add",
    name_prefix: str = "skip_cnn1d",
) -> tf.Tensor:
    """
    Apply projected skips for 3D tensors. This function can be used for convs 1D
    or graph layers that produce 3D outputs.

    Notes:
        Tensors are expected to have shape ``(batch, features, channels)``. The
        projection branch uses ``Conv1D(1)`` layers to adjust channels and, when
        possible, the temporal dimension. If the lengths still differ, a resize
        operation is applied.

    Args:
        trial: Optuna trial for selecting which skips to include.
        layers_list: Sequence of tensors from 1-D convolutional layers.
        axis_to_concat: Axis used for concatenation when ``merge_mode`` is
            ``"concat"``.
        use_batch_norm: Apply batch normalization in the projection branch.
        verbose: Verbosity level for skip combinations. ``0`` prints nothing,
            ``1`` prints only the number of combinations and ``2`` prints the
            number plus every combination. Defaults to ``1``.
        strategy: ``"final"`` or ``"any"`` to select candidate skip pairs.
        merge_mode: ``"add"`` or ``"concat"``.
        name_prefix: Prefix for projection layer names. Defaults to
            ``"skip_cnn1d"``.

    Returns:
        The merged output tensor after applying the selected skip connections.

    Raises:
        ValueError: If ``verbose`` not in ``{0, 1, 2}`` or if ``strategy`` or
            ``merge_mode`` are invalid.
    """

    return _trial_skip_connections_projected(
        trial=trial,
        layers_list=layers_list,
        project=lambda s, t, name: _project_conv1d(s, t, use_batch_norm, name),
        axis_to_concat=axis_to_concat,
        verbose=verbose,
        strategy=strategy,
        merge_mode=merge_mode,
        name_prefix=name_prefix,
    )


def trial_skip_2d_tensors(
    trial: optuna.trial.Trial,
    layers_list: Sequence[tf.Tensor],
    axis_to_concat: int = -1,
    use_batch_norm: bool = False,
    verbose: int = 1,
    strategy: str = "any",
    merge_mode: str = "add",
    name_prefix: str = "skip_dnn",
) -> tf.Tensor:
    """Skip connections for 2D tensors with feature projection.

    Notes:
        This function is designed for tensors with shape ``(batch, features)``.
        It applies a :class:`~keras.layers.Dense` layer to project the source
        tensor's features to match the target tensor's features. If the shapes
        differ, a resize operation is applied.

    Args:
        trial: Optuna trial controlling which connections are active.
        layers_list: Sequence of dense tensors of shape ``(batch, features)``.
        axis_to_concat: Axis used when concatenating outputs.
        use_batch_norm: Apply batch normalization in projection branches.
        verbose: Verbosity level for skip combinations. ``0`` prints nothing,
            ``1`` prints only the number of combinations and ``2`` prints the
            number plus every combination. Defaults to ``1``.
        strategy: ``"final"`` or ``"any"`` to define the skip topology.
        merge_mode: ``"add"`` or ``"concat"``.
        name_prefix: Prefix for projection layers, defaults to ``"skip_dnn"``.

    Returns:
        Output tensor after applying skip connections.

    Raises:
        ValueError: If ``verbose`` not in ``{0, 1, 2}`` or if ``strategy`` or
            ``merge_mode`` are invalid.
    """

    return _trial_skip_connections_projected(
        trial=trial,
        layers_list=layers_list,
        project=lambda s, t, name: _project_dense(s, t, use_batch_norm, name),
        axis_to_concat=axis_to_concat,
        verbose=verbose,
        strategy=strategy,
        merge_mode=merge_mode,
        name_prefix=name_prefix,
    )


def trial_skip_4d_tensors(
    trial: optuna.trial.Trial,
    layers_list: Sequence[tf.Tensor],
    axis_to_concat: int = -1,
    use_batch_norm: bool = False,
    verbose: int = 1,
    strategy: str = "any",
    merge_mode: str = "add",
    name_prefix: str = "skip_cnn2d",
) -> tf.Tensor:
    """Apply projected skip connections 4D tensors.

    Notes:
        This function is designed for tensors with shape ``(batch, height, width, channels)``.
        It applies a :class:`~keras.layers.Conv2D` layer with kernel size ``1``
        to project the source tensor's channels to match the target tensor's
        channels. If the spatial dimensions differ, a resize operation is applied.

    Args:
        trial: Optuna trial for selecting which skips to include.
        layers_list: Sequence of tensors produced by 2-D convolution layers with
            shape ``(batch, height, width, channels)``.
        axis_to_concat: Concatenation axis if ``merge_mode`` is ``"concat"``.
        use_batch_norm: Whether to apply batch normalization in the projection
            branch.
        verbose: Verbosity level for skip combinations. ``0`` prints nothing,
            ``1`` prints only the number of combinations and ``2`` prints the
            number plus every combination. Defaults to ``1``.
        strategy: ``"final"`` or ``"any"``.
        merge_mode: ``"add"`` or ``"concat"``.
        name_prefix: Prefix for projection layer names, defaults to
            ``"skip_cnn2d"``.

    Returns:
        The merged output tensor after applying the selected skip connections.

    Raises:
        ValueError: If ``verbose`` not in ``{0, 1, 2}`` or if ``strategy`` or
            ``merge_mode`` contain invalid values.
    """

    return _trial_skip_connections_projected(
        trial=trial,
        layers_list=layers_list,
        project=lambda s, t, name: _project_conv2d(s, t, use_batch_norm, name),
        axis_to_concat=axis_to_concat,
        verbose=verbose,
        strategy=strategy,
        merge_mode=merge_mode,
        name_prefix=name_prefix,
    )
