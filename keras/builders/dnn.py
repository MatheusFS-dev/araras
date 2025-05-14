from typing import *
from dataclasses import dataclass
from tensorflow.keras import layers, initializers
from araras.keras.hparams import HParams


@dataclass
class ResidualState:
    """
    Maintains state for residual connections during the construction of the DNN.

    Attributes:
        all_skips (List[layers.Layer]): A list of previous layers whose outputs
            can be used for residual (skip) connections. Initialized to an empty list
            after construction if not provided.
    """

    all_skips: List[layers.Layer] = None

    def __post_init__(self):
        # Ensures all_skips is initialized as an empty list if not provided.
        if self.all_skips is None:
            self.all_skips = []


def apply_residual(
    x: layers.Layer,
    trial: Any,
    idx: int,
    state: ResidualState,
    use_skip: bool,
    name_prefix: str,
) -> Tuple[layers.Layer, ResidualState]:
    """
    Applies optional residual (skip) connections from previous layers to the current layer.

    Logic flow:
        check use_skip ->
        if idx == 0: initialize skip list with current layer ->
        else:
            iterate over previous skips ->
                use optuna trial to decide whether to connect ->
                project if needed (match dimensions) ->
                add to collection ->
            if any connections made:
                apply keras Add to merge ->
            update skip list

    Args:
        x (layers.Layer): Current Keras layer to potentially apply residuals to.
        trial (Any): Optuna trial object used to sample boolean hyperparameters.
        idx (int): Current layer index in the DNN sequence.
        state (ResidualState): Object storing all previous skip connection candidates.
        use_skip (bool): Global flag indicating whether skip connections should be considered.
        name_prefix (str): String prefix used to uniquely name layers and parameters.

    Returns:
        Tuple[layers.Layer, ResidualState]: The updated layer and updated residual state.
    """
    if not use_skip:
        # Skip connections not enabled; return layer unchanged
        return x, state

    if idx == 0:
        # First layer, initialize residual state with the current layer
        state.all_skips = [x]
    else:
        to_add: List[layers.Layer] = []
        for j, prev in enumerate(state.all_skips):
            # Use Optuna to decide whether to include this previous layer in the skip connection
            if trial.suggest_categorical(f"{name_prefix}_use_res_{idx}_{j}", [True, False]):
                adj = prev
                # If output dimensions don't match, apply Dense layer to project dimensions
                if adj.shape[-1] != x.shape[-1]:
                    adj = layers.Dense(
                        units=x.shape[-1], activation=None, name=f"{name_prefix}_skip_proj_{idx}_{j}"
                    )(adj)
                # Collect valid skip connection
                to_add.append(adj)
        if to_add:
            # Add skip connections to the current layer using Add layer
            x = layers.Add(name=f"{name_prefix}_add_{idx}")([x] + to_add)
        # Add current layer to skip list for potential future connections
        state.all_skips.append(x)

    return x, state


def build_dnn(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    max_layers: int,
    max_units: int,
    min_units: int,
    units_step: int,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    min_dropout_rate: float = 0.0,
    max_dropout_rate: float = 0.5,
    dropout_rate_step: float = 0.1,
    dropout_positions: Optional[List[int]] = None,
    trial_batch_norm: bool = False,
    trial_regularizers: bool = False,
    regularizer_positions: Optional[List[int]] = None,
    trial_skip_connections: bool = False,
    share_activation: bool = False,
    name_prefix: str = "dnn",
) -> layers.Layer:
    """
    Builds a fully connected deep neural network with optional features like dropout,
    batch normalization, regularization, shared activations, and skip connections.

    Logic flow:
        initialize residual state ->
        optionally fetch shared activation ->
        for each layer:
            sample units ->
            get activation (shared or individual) ->
            optionally add kernel/bias/activity regularizers ->
            add Dense layer ->
            optionally add BatchNorm ->
            sample and apply Dropout ->
            optionally apply residual connections

    Args:
        trial (Any): Optuna trial used for hyperparameter sampling.
        hparams (HParams): Object that provides methods to fetch activation functions
            and regularizers based on trial and parameter name.
        x (layers.Layer): Input Keras layer to build upon.
        max_layers (int): Total number of layers to construct.
        max_units (int): Maximum number of units per Dense layer.
        min_units (int): Minimum number of units per Dense layer.
        units_step (int): Step size for unit range sampling.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for bias weights.
        min_dropout_rate (float): Minimum dropout rate.
        max_dropout_rate (float): Maximum dropout rate.
        dropout_rate_step (float): Step size for dropout rate sampling.
        dropout_positions (Optional[List[int]]): Specific layer indices for dropout to be applied. E.g. [0, 1]
        trial_batch_norm (bool): If True, batch normalization is considered.
        trial_regularizers (bool): If True, kernel/bias/activity regularizers are applied.
        regularizer_positions (Optional[List[int]]): Specific layer indices for regularizers to be applied. E.g. [0, 1]
        trial_skip_connections (bool): If True, residual connections are considered.
        share_activation (bool): If True, all layers use the same activation function.
        name_prefix (str): Prefix used for naming layers and hyperparameters.

    Returns:
        layers.Layer: Final Keras layer after building the DNN.
    """
    # Initialize state for managing skip connections across layers
    state = ResidualState()

    shared_act = None
    if share_activation:
        # If activations are shared, fetch the shared activation function
        shared_act = hparams.get_activation(trial, f"{name_prefix}_shared_act")

    for i in range(max_layers):
        # Sample number of units for this layer using Optuna
        units = trial.suggest_int(f"{name_prefix}_units_{i}", min_units, max_units, step=units_step)

        # Determine the activation function (shared or unique per layer)
        activation = shared_act or hparams.get_activation(trial, f"{name_prefix}_activation_{i}")

        # only sample regularizers if trial_regularizers and position matches
        if trial_regularizers and (regularizer_positions is None or i in regularizer_positions):
            kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg_{i}")
            bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg_{i}")
            act_reg = hparams.get_regularizer(trial, f"{name_prefix}_activity_reg_{i}")
        else:
            kernel_reg = bias_reg = act_reg = None

        # Add a Dense (fully connected) layer
        x = layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            name=f"{name_prefix}_layer_{i}",
        )(x)
    
        # Optionally add Batch Normalization layer if enabled and sampled as True
        if trial_batch_norm and trial.suggest_categorical(f"{name_prefix}_bn_{i}", [True, False]):
            x = layers.BatchNormalization(name=f"{name_prefix}_bn_{i}")(x)

        if dropout_positions is None or i in dropout_positions:
            # Sample dropout rate from defined range and apply Dropout layer
            dropout_rate = trial.suggest_float(
                f"{name_prefix}_dropout_{i}", min_dropout_rate, max_dropout_rate, step=dropout_rate_step
            )
            x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout_{i}")(x)

        # Apply residual (skip) connections if enabled
        x, state = apply_residual(x, trial, i, state, trial_skip_connections, name_prefix)

    return x
