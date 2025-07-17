"""
Last Edited: 14 July 2025
Description:
    Graph neural network layers and adjacency utilities.
"""

from araras.core import *

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers
from araras.ml.model.hyperparams import KParams
from spektral.layers import GCNConv, GATConv, ChebConv
import scipy.sparse as sp
from spektral.utils import convolution
from sklearn.neighbors import NearestNeighbors

PRINT_ONCE_JIT = True


def print_warning_jit():
    """Print a warning about JIT compilation."""
    global PRINT_ONCE_JIT
    if PRINT_ONCE_JIT:
        # warning messages in yellow
        warnings = [
            "==============================================================",
            "Spektral's GCNConv uses a sparse-dense matmul under the hood.",
            "XLA's GPU JIT compiler does not support that op.",
            "This may cause issues with the GNN layers.",
            "Disable all auto-JIT clustering and auto-JIT compilation.",
            "Call:",
        ]
        for msg in warnings:
            print(f"{YELLOW}{msg}{RESET}")

        # commands in orange
        commands = [
            "os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'",
            "tf.config.optimizer.set_jit(False)",
            "'jit_compile=False'  # pass into model.compile()",
        ]
        for cmd in commands:
            print(f"{ORANGE}{cmd}{RESET}")
        print(f"{YELLOW}=============================================================={RESET}")
        PRINT_ONCE_JIT = False


def build_grid_adjacency(rows: int, cols: int) -> tf.sparse.SparseTensor:
    """Build a grid adjacency matrix with GCN normalization.

    Each node is connected to its four direct neighbours (up, down, left and
    right).  The resulting adjacency matrix is returned as a TensorFlow sparse
    tensor ready to be fed to Spektral layers.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.

    Returns:
        tf.sparse.SparseTensor: Normalized sparse adjacency matrix.
    """
    n = rows * cols
    a = sp.lil_matrix((n, n), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    jdx = rr * cols + cc
                    a[idx, jdx] = 1.0

    a = a.tocsr()
    a = a + a.T
    a[a > 1] = 1.0

    a_norm = convolution.gcn_filter(a)
    coo = a_norm.tocoo()
    indices = np.vstack((coo.row, coo.col)).T
    return tf.sparse.reorder(tf.sparse.SparseTensor(indices, coo.data, coo.shape))


def build_knn_adjacency(rows: int, cols: int, k: int) -> tf.sparse.SparseTensor:
    """Construct a k-nearest neighbour adjacency matrix on a 2-D grid.

    Nodes correspond to cells of a `rows` × `cols` grid.  Each node is
    connected to its `k` spatially nearest neighbours.  The adjacency matrix is
    symmetrised, normalised with the GCN filter and returned as a TensorFlow
    sparse tensor.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.
        k: Number of neighbours for each node.

    Returns:
        tf.sparse.SparseTensor: Normalized sparse adjacency matrix.
    """

    n = rows * cols
    coords = np.array([(i // cols, i % cols) for i in range(n)], np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    a = sp.lil_matrix((n, n), dtype=np.float32)
    for i in range(n):
        for j in indices[i][1:]:
            a[i, j] = 1.0
            a[j, i] = 1.0
    a_norm = convolution.gcn_filter(a.tocsr())
    coo = a_norm.tocoo()
    idx = np.vstack((coo.row, coo.col)).T
    return tf.sparse.reorder(tf.sparse.SparseTensor(idx, coo.data, coo.shape))


def _select_range_value(
    trial: Any,
    name: str,
    value_range: Union[int, Tuple[int, int]],
    step: int,
) -> int:
    """Helper to pick an integer from a fixed value or an Optuna range."""
    if isinstance(value_range, int):
        return value_range
    low, high = value_range
    return trial.suggest_int(name, low, high, step=step)


def _select_float_range_value(
    trial: Any,
    name: str,
    value_range: Union[float, Tuple[float, float]],
    step: float,
) -> float:
    """Helper to pick a float from a fixed value or an Optuna range."""
    if isinstance(value_range, float):
        return value_range
    low, high = value_range
    return trial.suggest_float(name, low, high, step=step)


def _apply_layer_with_retry(
    layer: layers.Layer,
    inputs: list,
    name_prefix: str,
    retry_on_cpu: bool = False,
) -> tf.Tensor:
    """Execute a graph layer with optional CPU fallback.

    This helper centralises the handling of ``tf.errors.InvalidArgumentError``
    that arises when TensorFlow's GPU sparse--dense matrix multiplication
    exceeds its internal limits.  When ``retry_on_cpu`` is ``True`` the layer is
    re-executed on the CPU, otherwise the original exception is propagated.

    Args:
        layer: The Spektral convolutional layer instance to call.
        inputs: List of inputs ``[x, a_graph]`` for the layer.
        name_prefix: Prefix used when logging error messages.
        retry_on_cpu: If ``True``, retry the operation on the CPU on failure.

    Returns:
        The output tensor produced by ``layer``.

    Raises:
        tf.errors.InvalidArgumentError: If the GPU operation fails and
            ``retry_on_cpu`` is ``False``.
    """

    try:
        return layer(inputs)
    except tf.errors.InvalidArgumentError as exc:
        logger_error.fatal(
            f"{RED} Graph conv {name_prefix} hit GPU sparse-dense limit: {exc}{RESET}"
        )
        if retry_on_cpu:
            logger.info(f"{YELLOW}Retrying {name_prefix} on CPU...{RESET}")
            with tf.device("/CPU:0"):
                return layer(inputs)
        raise


def build_gcn(
    trial: Any,
    kparams: KParams,
    x: layers.Layer,
    a_graph: tf.sparse.SparseTensor,
    units_range: Union[int, Tuple[int, int]],
    dropout_rate_range: Union[float, Tuple[float, float]],
    units_step: int = 10,
    dropout_rate_step: float = 0.1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_bias: bool = True,
    use_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    retry_on_cpu: bool = False,
    name_prefix: str = "gcn",
) -> layers.Layer:
    """Build a single Graph Convolutional Network (GCN) layer.

    Creates a ``GCNConv`` layer and applies it to the input feature matrix and
    adjacency tensor.  The layer supports Optuna-based hyperparameter tuning for
    the number of units, dropout rate and regularizers.

    Warning:
        TensorFlow's GPU sparse--dense matrix multiplication has a hard limit
        ``output_channels * nnz(support) <= 2^31 - 1``.  When this limit is
        exceeded the operation fails with ``tf.errors.InvalidArgumentError``. Use
        ``retry_on_cpu=True`` to automatically rerun the layer on the CPU.

    Args:
        trial: Optuna trial object used for hyperparameter suggestions.
        kparams: Hyperparameter helper for activation functions and regularizers.
        x: Input feature tensor.
        a_graph: Normalized adjacency matrix as a sparse tensor.
        units_range: Either an integer or ``(low, high)`` tuple for the number of
            output units.
        dropout_rate_range: Float or ``(low, high)`` tuple for the dropout rate.
        units_step: Step size when ``units_range`` is a tuple.
        dropout_rate_step: Step size when ``dropout_rate_range`` is a tuple.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        use_bias: Whether to include a bias term.
        use_batch_norm: If ``True``, apply batch normalization after the layer.
        trial_kernel_reg: If ``True``, search for a kernel regularizer.
        trial_bias_reg: If ``True``, search for a bias regularizer.
        trial_activity_reg: If ``True``, search for an activity regularizer.
        retry_on_cpu: Retry the layer on the CPU when a GPU ``InvalidArgumentError`` occurs.
        name_prefix: Prefix for naming the created Keras layers.

    Returns:
        The output tensor after convolution, normalization, activation and dropout.

    Raises:
        tf.errors.InvalidArgumentError: If the GPU operation fails and
            ``retry_on_cpu`` is ``False``.
    """
    print_warning_jit()
    units = _select_range_value(trial, f"{name_prefix}_units", units_range, units_step)
    dropout = _select_float_range_value(
        trial, f"{name_prefix}_dropout", dropout_rate_range, dropout_rate_step
    )
    kernel_reg = kparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None
    bias_reg = kparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None
    act_reg = kparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    gnc_layer = GCNConv(
        channels=units,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        activity_regularizer=act_reg,
        name=name_prefix,
    )

    x = _apply_layer_with_retry(
        gnc_layer,
        [x, a_graph],
        name_prefix,
        retry_on_cpu=retry_on_cpu,
    )

    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    x = layers.Activation(kparams.get_activation(trial, f"{name_prefix}_act"), name=f"{name_prefix}_act")(x)

    x = layers.Dropout(dropout, name=f"{name_prefix}_dropout")(x)
    return x


def build_gat(
    trial: Any,
    kparams: KParams,
    x: layers.Layer,
    a_graph: tf.sparse.SparseTensor,
    units_range: Union[int, Tuple[int, int]],
    dropout_rate_range: Union[float, Tuple[float, float]],
    heads_range: Union[int, Tuple[int, int]],
    units_step: int = 10,
    dropout_rate_step: float = 0.1,
    heads_step: int = 1,
    concat_heads: bool = False,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_bias: bool = True,
    use_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    retry_on_cpu: bool = False,
    name_prefix: str = "gat",
) -> layers.Layer:
    """Build a single Graph Attention (GAT) layer.

    Applies ``GATConv`` to the input features with optional multi-head
    attention. Hyperparameters such as units, number of heads and dropout rate
    can be tuned via Optuna trials.

    Warning:
        As with ``build_gcn``, the GPU kernel may fail when
        ``output_channels * nnz(support)`` exceeds ``2^31 - 1``. Enable
        ``retry_on_cpu`` to fall back to a CPU implementation in this case.

    Args:
        trial: Optuna trial object for hyperparameter sampling.
        kparams: Helper providing activations and regularizers.
        x: Input feature tensor.
        a_graph: Normalized adjacency matrix as a sparse tensor.
        units_range: Integer or ``(low, high)`` tuple specifying output units.
        dropout_rate_range: Float or ``(low, high)`` tuple for dropout.
        heads_range: Integer or ``(low, high)`` tuple for the number of heads.
        units_step: Step size when sampling ``units_range``.
        dropout_rate_step: Step size when sampling ``dropout_rate_range``.
        heads_step: Step size when sampling ``heads_range``.
        concat_heads: Concatenate the outputs of the attention heads if ``True``.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        use_bias: Whether to include a bias term.
        use_batch_norm: If ``True``, apply batch normalization after the layer.
        trial_kernel_reg: If ``True``, search for a kernel regularizer.
        trial_bias_reg: If ``True``, search for a bias regularizer.
        trial_activity_reg: If ``True``, search for an activity regularizer.
        retry_on_cpu: Retry the operation on the CPU if a GPU
            ``InvalidArgumentError`` is raised.
        name_prefix: Prefix for naming the created Keras layers.

    Returns:
        The output tensor after convolution, normalization, activation and dropout.

    Raises:
        tf.errors.InvalidArgumentError: If the GPU operation fails and
            ``retry_on_cpu`` is ``False``.
    """
    print_warning_jit()
    units = _select_range_value(trial, f"{name_prefix}_units", units_range, units_step)
    heads = _select_range_value(trial, f"{name_prefix}_heads", heads_range, heads_step)
    dropout = _select_float_range_value(
        trial, f"{name_prefix}_dropout", dropout_rate_range, dropout_rate_step
    )

    kernel_reg = kparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None
    bias_reg = kparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None
    act_reg = kparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    gat_layer = GATConv(
        channels=units,
        activation=None,
        attn_heads=heads,
        concat=concat_heads,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        activity_regularizer=act_reg,
        name=name_prefix,
    )

    x = _apply_layer_with_retry(
        gat_layer,
        [x, a_graph],
        name_prefix,
        retry_on_cpu=retry_on_cpu,
    )

    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    x = layers.Activation(kparams.get_activation(trial, f"{name_prefix}_act"), name=f"{name_prefix}_act")(x)

    x = layers.Dropout(dropout, name=f"{name_prefix}_dropout")(x)
    return x


def build_cheb(
    trial: Any,
    kparams: KParams,
    x: layers.Layer,
    a_graph: tf.sparse.SparseTensor,
    units_range: Union[int, Tuple[int, int]],
    dropout_rate_range: Union[float, Tuple[float, float]],
    K_range: Union[int, Tuple[int, int]],
    units_step: int = 10,
    dropout_rate_step: float = 0.1,
    K_step: int = 1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_bias: bool = True,
    use_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    retry_on_cpu: bool = False,
    name_prefix: str = "cheb",
) -> layers.Layer:
    """Build a single Chebyshev graph convolution layer.

    Applies ``ChebConv`` using a Chebyshev polynomial approximation of the graph
    Laplacian. The polynomial order ``K`` along with the number of units and
    dropout rate can be tuned via Optuna.

    Warning:
        The GPU sparse--dense kernel has the same ``2^31 - 1`` limitation as in
        ``build_gcn`` and ``build_gat``. Use ``retry_on_cpu`` to execute the
        layer on the CPU when this error occurs.

    Args:
        trial: Optuna trial used for suggesting hyperparameters.
        kparams: Hyperparameter helper instance.
        x: Input feature tensor.
        a_graph: Normalized adjacency matrix as a sparse tensor.
        units_range: Integer or ``(low, high)`` tuple for output units.
        dropout_rate_range: Float or ``(low, high)`` tuple for dropout rate.
        K_range: Integer or ``(low, high)`` tuple for the Chebyshev order ``K``.
        units_step: Step size when sampling ``units_range``.
        dropout_rate_step: Step size when sampling ``dropout_rate_range``.
        K_step: Step size when sampling ``K_range``.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        use_bias: Whether to include a bias term.
        use_batch_norm: If ``True``, apply batch normalization after the layer.
        trial_kernel_reg: If ``True``, search for a kernel regularizer.
        trial_bias_reg: If ``True``, search for a bias regularizer.
        trial_activity_reg: If ``True``, search for an activity regularizer.
        retry_on_cpu: Retry the operation on the CPU if a GPU
            ``InvalidArgumentError`` is raised.
        name_prefix: Prefix for naming the created Keras layers.

    Returns:
        The output tensor after convolution, normalization, activation and dropout.

    Raises:
        tf.errors.InvalidArgumentError: If the GPU operation fails and
            ``retry_on_cpu`` is ``False``.
    """
    print_warning_jit()
    units = _select_range_value(trial, f"{name_prefix}_units", units_range, units_step)
    K = _select_range_value(trial, f"{name_prefix}_K", K_range, K_step)
    dropout = _select_float_range_value(
        trial, f"{name_prefix}_dropout", dropout_rate_range, dropout_rate_step
    )

    kernel_reg = kparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None
    bias_reg = kparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None
    act_reg = kparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    cheb_layer = ChebConv(
        channels=units,
        K=K,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        activity_regularizer=act_reg,
        name=name_prefix,
    )

    x = _apply_layer_with_retry(
        cheb_layer,
        [x, a_graph],
        name_prefix,
        retry_on_cpu=retry_on_cpu,
    )

    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    x = layers.Activation(kparams.get_activation(trial, f"{name_prefix}_act"), name=f"{name_prefix}_act")(x)

    x = layers.Dropout(dropout, name=f"{name_prefix}_dropout")(x)
    return x
