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
                    adj = layers.Conv1DTranspose(
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


def build_tcnn1d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    max_layers: int,
    max_filters: int,
    min_filters: int,
    filters_step: int,
    max_kernel_size: int,
    min_kernel_size: int,
    min_upsample_size: int,
    max_upsample_size: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    trial_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    regularizer_positions: Optional[List[int]] = None,
    trial_skip_connections: bool = False,
    share_activation: bool = False,
    name_prefix: str = "tcnn1d",
) -> layers.Layer:
    """
    Constructs a 1D Transpose Convolutional Neural Network with optional skip connections,
    batch normalization, and regularization using hyperparameter optimization.

    Logic Flow:
        share_activation=True -> get shared activation
        for i in max_layers:
            -> get filters, kernel_size, pool_size from trial
            -> get activation (shared or per-layer)
            -> conditionally apply regularizers
            -> build Conv1DTranspose layer
            -> conditionally apply BatchNorm
            -> optionally apply skip connections
            -> apply UpSampling1D

    Args:
        trial (Any): Optimization trial object.
        hparams (HParams): Hyperparameter helper object.
        x (layers.Layer): Input tensor.
        max_layers (int): Number of Conv1DTranspose blocks to build.
        max_filters (int): Maximum number of filters in Conv1DTranspose.
        min_filters (int): Minimum number of filters in Conv1DTranspose.
        filters_step (int): Step size for filter count.
        max_kernel_size (int): Maximum kernel size.
        min_kernel_size (int): Minimum kernel size.
        min_upsample_size (int): Minimum upsampling size.
        max_upsample_size (int): Maximum upsampling size.
        data_format (str): Tensor data format.
        padding (str): Padding strategy.
        strides (int): Stride length.
        dilation_rate (int): Dilation rate for Conv1DTranspose.
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
        layers.Layer: Final output tensor after applying Conv1DTranspose layers and pooling.
    """
    state = ResidualState()
    shared_act = None

    # If sharing activations across all layers, retrieve once from hparams
    if share_activation:
        shared_act = hparams.get_activation(trial, f"{name_prefix}_activation")

    for i in range(max_layers):
        # Dynamically suggest filters, kernel size, and pooling size
        filters = trial.suggest_int(f"{name_prefix}_filters_{i}", min_filters, max_filters, step=filters_step)
        size = trial.suggest_int(f"{name_prefix}_pool_size_{i}", min_upsample_size, max_upsample_size)
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

        # Apply Conv1DTranspose layer with specified parameters
        x = layers.Conv1DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            data_format=data_format,
            strides=strides,
            dilation_rate=dilation_rate,
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

        # Apply UpSampling1D
        x = layers.UpSampling1D(size=size, name=f"{name_prefix}_upsample_{i}")(x)

    return x