from typing import *
from dataclasses import dataclass
from tensorflow.keras import layers, initializers
from araras.keras.hparams import HParams


@dataclass
class ResidualState:
    """
    Tracks the state of residual connections throughout a network's layers.

    Attributes:
        all_skips (List[layers.Layer]): List of layers used as skip connections.
    """

    all_skips: List[layers.Layer] = None

    def __post_init__(self):
        # Initialize all_skips as an empty list if not provided
        if self.all_skips is None:
            self.all_skips = []


def apply_residual_1d(
    x: layers.Layer,
    trial: Any,
    idx: int,
    state: ResidualState,
    use_skip: bool,
    name_prefix: str,
) -> Tuple[layers.Layer, ResidualState]:
    """
    Applies optional 1D residual (skip) connections to a layer.

    Logic Flow:
        use_skip=False -> return x
        idx == 0 -> store x as first skip
        else -> for each previous skip:
            -> optionally project to correct shape with Conv1D
            -> add to current layer
        -> store updated x for future skips

    Args:
        x (layers.Layer): Current layer output.
        trial (Any): Optimization trial object for hyperparameter suggestions.
        idx (int): Index of the current layer.
        state (ResidualState): Tracks residual connections.
        use_skip (bool): Whether to use residual connections.
        name_prefix (str): Name prefix for naming layers.

    Returns:
        Tuple[layers.Layer, ResidualState]: Updated layer and state.
    """
    if not use_skip:
        return x, state

    if idx == 0:
        # At first layer, simply store the input as the initial skip
        state.all_skips = [x]
    else:
        to_add: List[layers.Layer] = []
        # Iterate through previous skip connections
        for j, prev in enumerate(state.all_skips):
            if trial.suggest_categorical(f"{name_prefix}_use_res_{idx}_{j}", [True, False]):
                adj = prev
                if adj.shape[-1] != x.shape[-1]:
                    # Adjust channels via 1x1 Conv1D if necessary
                    adj = layers.Conv1D(
                        filters=x.shape[-1],
                        kernel_size=1,
                        padding="same",
                        name=f"{name_prefix}_skip_proj_{idx}_{j}",
                    )(adj)
                to_add.append(adj)
        if to_add:
            # Combine the current layer with all valid skip connections
            x = layers.Add(name=f"{name_prefix}_add_{idx}")([x] + to_add)
        # Store the current output for future residuals
        state.all_skips.append(x)

    return x, state


def build_cnn1d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    max_layers: int,
    max_filters: int,
    min_filters: int,
    filters_step: int,
    max_kernel_size: int,
    min_kernel_size: int,
    min_pool_size: int,
    max_pool_size: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    groups: int = 1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    trial_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    regularizer_positions: Optional[List[int]] = None,
    trial_skip_connections: bool = False,
    share_activation: bool = False,
    name_prefix: str = "cnn1d",
) -> layers.Layer:
    """
    Constructs a 1D Convolutional Neural Network with optional skip connections,
    batch normalization, and regularization using hyperparameter optimization.

    Logic Flow:
        share_activation=True -> get shared activation
        for i in max_layers:
            -> get filters, kernel_size, pool_size from trial
            -> get activation (shared or per-layer)
            -> conditionally apply regularizers
            -> build Conv1D layer
            -> conditionally apply BatchNorm
            -> optionally apply skip connections
            -> apply MaxPooling1D

    Args:
        trial (Any): Optimization trial object.
        hparams (HParams): Hyperparameter helper object.
        x (layers.Layer): Input tensor.
        max_layers (int): Number of Conv1D blocks to build.
        max_filters (int): Maximum number of filters in Conv1D.
        min_filters (int): Minimum number of filters in Conv1D.
        filters_step (int): Step size for filter count.
        max_kernel_size (int): Maximum kernel size.
        min_kernel_size (int): Minimum kernel size.
        min_pool_size (int): Minimum pooling size.
        max_pool_size (int): Maximum pooling size.
        data_format (str): Tensor data format.
        padding (str): Padding strategy.
        strides (int): Stride length.
        dilation_rate (int): Dilation rate for Conv1D.
        groups (int): Number of groups for grouped convolution.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for bias.
        trial_batch_norm (bool): Enable batch normalization decision by trial.
        trial_kernel_reg (bool): Enable kernel regularization.
        trial_bias_reg (bool): Enable bias regularization.
        trial_activity_reg (bool): Enable activity regularization.
        regularizer_positions (Optional[List[int]]): Layer indices to apply regularizers.
        trial_skip_connections (bool): Enable residual connections.
        share_activation (bool): Share activation function across all layers.
        name_prefix (str): Prefix for naming layers.

    Returns:
        layers.Layer: Final output tensor after applying Conv1D layers and pooling.
    """
    state = ResidualState()
    shared_act = None

    # If sharing activations across all layers, retrieve once from hparams
    if share_activation:
        shared_act = hparams.get_activation(trial, f"{name_prefix}_activation")

    for i in range(max_layers):
        # Dynamically suggest filters, kernel size, and pooling size
        filters = trial.suggest_int(f"{name_prefix}_filters_{i}", min_filters, max_filters, step=filters_step)
        pool_size = trial.suggest_int(f"{name_prefix}_pool_size_{i}", min_pool_size, max_pool_size)
        kernel_size = trial.suggest_int(f"{name_prefix}_kernel_size_{i}", min_kernel_size, max_kernel_size)

        # Use shared activation if specified, otherwise layer-specific
        activation = shared_act or hparams.get_activation(trial, f"{name_prefix}_activation_{i}")

        # Conditionally apply regularizers based on flags and allowed positions
        kernel_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg_{i}") if trial_kernel_reg else None
        )
        bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg_{i}") if trial_bias_reg else None
        act_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_activity_reg_{i}") if trial_activity_reg else None
        )
        if regularizer_positions is not None and i not in regularizer_positions:
            kernel_reg = bias_reg = act_reg = None

        # Apply Conv1D layer with specified parameters
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            data_format=data_format,
            strides=strides,
            dilation_rate=dilation_rate,
            groups=groups,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            name=f"{name_prefix}_{i}",
        )(x)

        # Apply optional BatchNormalization
        if trial_batch_norm and trial.suggest_categorical(f"{name_prefix}_bn_{i}", [True, False]):
            x = layers.BatchNormalization(name=f"{name_prefix}_bn_{i}")(x)

        # Optionally apply residual connections
        x, state = apply_residual_1d(x, trial, i, state, trial_skip_connections, name_prefix)

        # Apply MaxPooling1D
        x = layers.MaxPooling1D(pool_size=pool_size, name=f"{name_prefix}_pool_{i}")(x)

    return x


def apply_residual_2d(
    x: layers.Layer,
    trial: Any,
    idx: int,
    state: ResidualState,
    use_skip: bool,
    name_prefix: str,
) -> Tuple[layers.Layer, ResidualState]:
    """
    Applies optional 2D residual (skip) connections from previous layers.
    Uses Conv2D with 1x1 kernels to project channels if shapes mismatch.

    Logic Flow:
        use_skip=False -> return x
        idx==0 -> store x
        else:
            -> iterate over previous skips
            -> for each: sample whether to use it
                -> if shapes mismatch: apply 1x1 Conv2D
                -> collect to_add
            -> Add all valid residuals to x
            -> Store x in skips

    Args:
        x (layers.Layer): Current 2D tensor output.
        trial (Any): Optimization trial object.
        idx (int): Index of the current layer.
        state (ResidualState): Residual connection tracker.
        use_skip (bool): Whether to apply skip connections.
        name_prefix (str): Prefix for layer names.

    Returns:
        Tuple[layers.Layer, ResidualState]: Updated tensor and residual state.
    """
    if not use_skip:
        return x, state

    if idx == 0:
        # Save initial input as skip connection for first layer
        state.all_skips = [x]
    else:
        to_add: List[layers.Layer] = []
        # Loop through previous skip connections
        for j, prev in enumerate(state.all_skips):
            if trial.suggest_categorical(f"{name_prefix}_use_res_{idx}_{j}", [True, False]):
                adj = prev
                # Match channel depth via Conv2D(1x1) if necessary
                if adj.shape[-1] != x.shape[-1]:
                    adj = layers.Conv2D(
                        filters=x.shape[-1],
                        kernel_size=(1, 1),
                        padding="same",
                        name=f"{name_prefix}_skip_proj_{idx}_{j}",
                    )(adj)
                to_add.append(adj)
        if to_add:
            # Add current tensor to the selected skip connections
            x = layers.Add(name=f"{name_prefix}_add_{idx}")([x] + to_add)
        # Store updated output for future skip use
        state.all_skips.append(x)

    return x, state


def build_cnn2d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    max_layers: int,
    max_filters: int,
    min_filters: int,
    filters_step: int,
    max_kernel_size: int,
    min_kernel_size: int,
    min_pool_size: int,
    max_pool_size: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: Tuple[int, int] = (1, 1),
    dilation_rate: Tuple[int, int] = (1, 1),
    groups: int = 1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    trial_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    regularizer_positions: Optional[List[int]] = None,
    trial_skip_connections: bool = False,
    share_activation: bool = False,
    name_prefix: str = "cnn2d",
) -> layers.Layer:
    """
    Builds a 2D CNN composed of Conv2D layers with optional skip connections,
    batch normalization, and regularizers, driven by hyperparameter tuning.

    Logic Flow:
        share_activation=True -> retrieve shared activation
        for i in range(max_layers):
            -> determine filters, kernel, pool sizes
            -> choose activation (shared or unique)
            -> conditionally assign regularizers
            -> apply Conv2D
            -> optional BatchNorm
            -> optional residuals
            -> apply MaxPooling2D

    Args:
        trial (Any): Trial object for Optuna-style parameter sampling.
        hparams (HParams): Hyperparameter utility for retrieving activations and regularizers.
        x (layers.Layer): Input tensor.
        max_layers (int): Number of convolutional layers.
        max_filters (int): Maximum number of filters.
        min_filters (int): Minimum number of filters.
        filters_step (int): Step size for filters.
        max_kernel_size (int): Max size of kernel dimension.
        min_kernel_size (int): Min size of kernel dimension.
        min_pool_size (int): Minimum pooling window size.
        max_pool_size (int): Maximum pooling window size.
        data_format (str): Tensor format (channels_last or channels_first).
        padding (str): Padding strategy.
        strides (Tuple[int, int]): Convolution stride.
        dilation_rate (Tuple[int, int]): Dilation rate.
        groups (int): Number of convolution groups.
        kernel_initializer (initializers.Initializer): Kernel initializer.
        bias_initializer (initializers.Initializer): Bias initializer.
        trial_batch_norm (bool): Enable batch norm based on trial.
        trial_kernel_reg (bool): Enable kernel regularization based on trial.
        trial_bias_reg (bool): Enable bias regularization based on trial.
        trial_activity_reg (bool): Enable activity regularization based on trial.
        regularizer_positions (Optional[List[int]]): List of layer indices to apply regularizers.
        trial_skip_connections (bool): Whether to include residual connections.
        share_activation (bool): Use one shared activation function across all layers.
        name_prefix (str): Prefix to use when naming layers.

    Returns:
        layers.Layer: Output tensor after applying all CNN layers and pooling.
    """
    state = ResidualState()
    shared_act = None

    if share_activation:
        shared_act = hparams.get_activation(trial, f"{name_prefix}_activation")

    for i in range(max_layers):
        # Suggest hyperparameters for the current layer
        filters = trial.suggest_int(f"{name_prefix}_filters_{i}", min_filters, max_filters, step=filters_step)
        pool_dim1 = trial.suggest_int(f"{name_prefix}_pool_dim1_{i}", min_pool_size, max_pool_size)
        pool_dim2 = trial.suggest_int(f"{name_prefix}_pool_dim2_{i}", min_pool_size, max_pool_size)
        pool_size = (pool_dim1, pool_dim2)

        kernel_dim1 = trial.suggest_int(f"{name_prefix}_kernel_dim1_{i}", min_kernel_size, max_kernel_size)
        kernel_dim2 = trial.suggest_int(f"{name_prefix}_kernel_dim2_{i}", min_kernel_size, max_kernel_size)
        kernel_size = (kernel_dim1, kernel_dim2)

        # Use shared or layer-specific activation
        activation = shared_act or hparams.get_activation(trial, f"{name_prefix}_activation_{i}")

        # Assign regularizers if applicable
        kernel_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg_{i}") if trial_kernel_reg else None
        )
        bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg_{i}") if trial_bias_reg else None
        act_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_activity_reg_{i}") if trial_activity_reg else None
        )
        if regularizer_positions is not None and i not in regularizer_positions:
            kernel_reg = bias_reg = act_reg = None

        # Apply Conv2D
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            data_format=data_format,
            strides=strides,
            dilation_rate=dilation_rate,
            groups=groups,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            name=f"{name_prefix}_{i}",
        )(x)

        # Conditionally apply batch normalization
        if trial_batch_norm and trial.suggest_categorical(f"{name_prefix}_bn_{i}", [True, False]):
            x = layers.BatchNormalization(name=f"{name_prefix}_bn_{i}")(x)

        # Apply residual connections if enabled
        x, state = apply_residual_2d(x, trial, i, state, trial_skip_connections, name_prefix)

        # Apply max pooling
        x = layers.MaxPooling2D(pool_size=pool_size, data_format=data_format, name=f"{name_prefix}_pool_{i}")(
            x
        )

    return x


def apply_residual_3d(
    x: layers.Layer,
    trial: Any,
    idx: int,
    state: ResidualState,
    use_skip: bool,
    name_prefix: str,
) -> Tuple[layers.Layer, ResidualState]:
    """
    Applies optional 3D residual (skip) connections from earlier layers to the current layer.
    Ensures channel compatibility via 1×1×1 Conv3D projection if necessary.

    Logic Flow:
        use_skip=False -> return x
        idx==0 -> store x
        else:
            -> for each skip:
                -> decide to use or not
                -> match channels via Conv3D(1x1x1) if needed
            -> add selected skips to x
            -> store updated x

    Args:
        x (layers.Layer): Output tensor from the current layer.
        trial (Any): Hyperparameter tuning object.
        idx (int): Index of the current layer.
        state (ResidualState): Skip connection tracker.
        use_skip (bool): Whether to use skip connections.
        name_prefix (str): Name prefix for layers.

    Returns:
        Tuple[layers.Layer, ResidualState]: Updated tensor and updated state.
    """
    if not use_skip:
        return x, state

    if idx == 0:
        state.all_skips = [x]
    else:
        to_add: List[layers.Layer] = []
        for j, prev in enumerate(state.all_skips):
            if trial.suggest_categorical(f"{name_prefix}_use_res_{idx}_{j}", [True, False]):
                adj = prev
                if adj.shape[-1] != x.shape[-1]:
                    adj = layers.Conv3D(
                        filters=x.shape[-1],
                        kernel_size=(1, 1, 1),
                        padding="same",
                        name=f"{name_prefix}_skip_proj_{idx}_{j}",
                    )(adj)
                to_add.append(adj)
        if to_add:
            x = layers.Add(name=f"{name_prefix}_add_{idx}")([x] + to_add)
        state.all_skips.append(x)

    return x, state


def build_cnn3d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    max_layers: int,
    max_filters: int,
    min_filters: int,
    filters_step: int,
    max_kernel_size: int,
    min_kernel_size: int,
    min_pool_size: int,
    max_pool_size: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: Tuple[int, int, int] = (1, 1, 1),
    dilation_rate: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    trial_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    regularizer_positions: Optional[List[int]] = None,
    trial_skip_connections: bool = False,
    share_activation: bool = False,
    name_prefix: str = "cnn3d",
) -> layers.Layer:
    """
    Builds a 3D CNN using Conv3D layers, optionally adding batch normalization,
    skip connections, and various regularizers, controlled by hyperparameter trials.

    Logic Flow:
        if share_activation:
            -> get shared activation
        for i in range(max_layers):
            -> suggest filters, kernel, pool sizes
            -> choose activation
            -> get optional regularizers
            -> build Conv3D
            -> optionally apply batch norm
            -> optionally add skip connection
            -> apply MaxPooling3D

    Args:
        trial (Any): Hyperparameter tuning trial.
        hparams (HParams): Hyperparameter utility instance.
        x (layers.Layer): Input tensor.
        max_layers (int): Maximum number of Conv3D layers.
        max_filters (int): Upper bound for filter count.
        min_filters (int): Lower bound for filter count.
        filters_step (int): Step size for filters.
        max_kernel_size (int): Maximum kernel dimension.
        min_kernel_size (int): Minimum kernel dimension.
        min_pool_size (int): Minimum pooling size.
        max_pool_size (int): Maximum pooling size.
        data_format (str): Format of input tensor.
        padding (str): Padding strategy.
        strides (Tuple[int, int, int]): Strides for convolution.
        dilation_rate (Tuple[int, int, int]): Dilation rates.
        groups (int): Number of groups in grouped convolution.
        kernel_initializer (initializers.Initializer): Kernel initializer.
        bias_initializer (initializers.Initializer): Bias initializer.
        trial_batch_norm (bool): Whether to sample batch norm use.
        trial_kernel_reg (bool): Whether to include kernel regularization.
        trial_bias_reg (bool): Whether to include bias regularization.
        trial_activity_reg (bool): Whether to include activity regularization.
        regularizer_positions (Optional[List[int]]): Specific layers to apply regularization.
        trial_skip_connections (bool): Enable skip connections.
        share_activation (bool): Share activation across layers.
        name_prefix (str): Prefix used in naming layers.

    Returns:
        layers.Layer: Output tensor after all Conv3D blocks and pooling.
    """
    state = ResidualState()
    shared_act = None

    if share_activation:
        shared_act = hparams.get_activation(trial, f"{name_prefix}_activation")

    for i in range(max_layers):
        filters = trial.suggest_int(f"{name_prefix}_filters_{i}", min_filters, max_filters, step=filters_step)

        # Sample pooling dimensions
        pd1 = trial.suggest_int(f"{name_prefix}_pool_dim1_{i}", min_pool_size, max_pool_size)
        pd2 = trial.suggest_int(f"{name_prefix}_pool_dim2_{i}", min_pool_size, max_pool_size)
        pd3 = trial.suggest_int(f"{name_prefix}_pool_dim3_{i}", min_pool_size, max_pool_size)
        pool_size = (pd1, pd2, pd3)

        # Sample kernel dimensions
        kd1 = trial.suggest_int(f"{name_prefix}_kernel_dim1_{i}", min_kernel_size, max_kernel_size)
        kd2 = trial.suggest_int(f"{name_prefix}_kernel_dim2_{i}", min_kernel_size, max_kernel_size)
        kd3 = trial.suggest_int(f"{name_prefix}_kernel_dim3_{i}", min_kernel_size, max_kernel_size)
        kernel_size = (kd1, kd2, kd3)

        activation = shared_act or hparams.get_activation(trial, f"{name_prefix}_activation_{i}")

        kernel_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg_{i}") if trial_kernel_reg else None
        )
        bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg_{i}") if trial_bias_reg else None
        act_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_activity_reg_{i}") if trial_activity_reg else None
        )
        if regularizer_positions is not None and i not in regularizer_positions:
            kernel_reg = bias_reg = act_reg = None

        x = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            data_format=data_format,
            strides=strides,
            dilation_rate=dilation_rate,
            groups=groups,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            name=f"{name_prefix}_{i}",
        )(x)

        if trial_batch_norm and trial.suggest_categorical(f"{name_prefix}_bn_{i}", [True, False]):
            x = layers.BatchNormalization(name=f"{name_prefix}_bn_{i}")(x)

        x, state = apply_residual_3d(x, trial, i, state, trial_skip_connections, name_prefix)

        x = layers.MaxPooling3D(pool_size=pool_size, data_format=data_format, name=f"{name_prefix}_pool_{i}")(
            x
        )

    return x
