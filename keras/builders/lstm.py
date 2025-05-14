from typing import *
from dataclasses import dataclass
from tensorflow.keras import layers, initializers
from araras.keras.hparams import HParams

# TODO: implement all features from https://keras.io/api/layers/recurrent_layers/lstm/

@dataclass
class ResidualState:
    """
    Maintains state for residual connections during stacking of layers.

    Attributes:
        all_skips (List[layers.Layer]): List of previous layer outputs eligible
            for skip connections. Initialized to an empty list if not provided.
    """

    all_skips: List[layers.Layer] = None

    def __post_init__(self):
        # Ensure we always have a list to append to
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
    Applies optional residual (skip) connections from earlier layers to the current layer.

    Logic flow:
      1. If skip connections are disabled, return immediately.
      2. If this is the first layer (idx == 0), initialize state.all_skips.
      3. Otherwise:
         a. For each prior layer in state.all_skips, use Optuna to decide inclusion.
         b. If chosen and dimensions differ, project with a Dense layer.
         c. Collect all selected skips.
         d. If any, merge them with the current output via layers.Add.
         e. Append the (possibly merged) output to state.all_skips for future layers.

    Args:
        x (layers.Layer): The current layer output to augment.
        trial (Any): Optuna trial for sampling boolean choices.
        idx (int): Index of the current layer in the stack.
        state (ResidualState): Holds previous layers for skip candidates.
        use_skip (bool): Global flag enabling skip logic.
        name_prefix (str): Prefix for naming skip-related layers and params.

    Returns:
        Tuple[layers.Layer, ResidualState]: The (potentially) modified layer
        and updated residual state.
    """
    if not use_skip:
        # Skip connections globally disabled
        return x, state

    if idx == 0:
        # First layer: seed the skip list
        state.all_skips = [x]
    else:
        to_add: List[layers.Layer] = []
        for j, prev in enumerate(state.all_skips):
            # Decide per-pair whether to connect
            if trial.suggest_categorical(f"{name_prefix}_use_res_{idx}_{j}", [True, False]):
                adj = prev
                # Project dimensions if they don't match
                if adj.shape[-1] != x.shape[-1]:
                    adj = layers.Dense(
                        units=x.shape[-1], activation=None, name=f"{name_prefix}_skip_proj_{idx}_{j}"
                    )(adj)
                to_add.append(adj)

        if to_add:
            # Merge current output with all selected skips
            x = layers.Add(name=f"{name_prefix}_add_{idx}")([x] + to_add)

        # Make this layer eligible for future skips
        state.all_skips.append(x)

    return x, state


def build_lstm(
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
    trial_recurrent_dropout: bool = False,
    trial_dropout: bool = False,
    trial_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    regularizer_positions: Optional[List[int]] = None,
    trial_skip_connections: bool = False,
    share_activation: bool = False,
    name_prefix: str = "lstm",
) -> layers.Layer:
    """
    Builds a stack of LSTM layers with hyperparameter-driven internal dropout,
    batch normalization, regularization, shared activation, and skip connections.

    Logic flow:
      1. Initialize residual state
      2. Optionally sample a shared activation
      3. For each layer index i in [0..max_layers):
         a. Sample units
         b. Determine activation (shared or per-layer)
         c. Sample/apply kernel, bias, and activity regularizers (if enabled & in position)
         d. Sample internal dropout and recurrent_dropout rates (if enabled)
         e. Set return_sequences=True for all but the final layer
         f. Add the LSTM layer
         g. Optionally add BatchNormalization
         h. Apply residual (skip) connections

    Args:
        trial: Optuna trial for hyperparameter sampling.
        hparams: Provides activation and regularizer factories.
        x: Input Keras layer.
        max_layers: Number of LSTM layers.
        max_units: Upper bound for units in each layer.
        min_units: Lower bound for units in each layer.
        units_step: Step size for unit sampling.
        kernel_initializer: Initializer for the LSTM kernel.
        bias_initializer: Initializer for the LSTM bias.
        min_dropout_rate: Minimum internal dropout rate.
        max_dropout_rate: Maximum internal dropout rate.
        dropout_rate_step: Step size for dropout rate sampling.
        trial_recurrent_dropout: If True, sample recurrent_dropout per layer.
        trial_dropout: If True, sample dropout per layer.
        trial_batch_norm: If True, consider batch normalization after each layer.
        trial_kernel_reg: If True, consider kernel regularizers.
        trial_bias_reg: If True, consider bias regularizers.
        trial_activity_reg: If True, consider activity regularizers.
        regularizer_positions: Indices at which to apply regularizers.
        trial_skip_connections: If True, allow residual connections.
        share_activation: If True, use one shared activation for all layers.
        name_prefix: Prefix for naming layers and hyperparameters.

    Returns:
        The final Keras layer after stacking all LSTMs.
    """
    # initialize residual state
    state = ResidualState()

    # sample a shared activation if requested
    shared_act = None
    if share_activation:
        shared_act = hparams.get_activation(trial, f"{name_prefix}_activation")

    for i in range(max_layers):
        # sample number of units
        units = trial.suggest_int(f"{name_prefix}_units_{i}", min_units, max_units, step=units_step)

        # determine activation
        activation = shared_act or hparams.get_activation(trial, f"{name_prefix}_activation_{i}")

        # sample regularizers if enabled and in position
        kernel_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg_{i}") if trial_kernel_reg else None
        )
        bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg_{i}") if trial_bias_reg else None
        act_reg = (
            hparams.get_regularizer(trial, f"{name_prefix}_activity_reg_{i}") if trial_activity_reg else None
        )
        if regularizer_positions is not None and i not in regularizer_positions:
            kernel_reg = bias_reg = act_reg = None

        # sample internal dropout rates
        dropout_rate = (
            trial.suggest_float(
                f"{name_prefix}_dropout_{i}",
                min_dropout_rate,
                max_dropout_rate,
                step=dropout_rate_step,
            )
            if trial_dropout
            else 0.0
        )
        recurrent_dropout = (
            trial.suggest_float(
                f"{name_prefix}_recurrent_dropout_{i}",
                min_dropout_rate,
                max_dropout_rate,
                step=dropout_rate_step,
            )
            if trial_recurrent_dropout
            else 0.0
        )

        # use return_sequences on all but the last layer
        return_seq = i < max_layers - 1

        # add the LSTM layer
        x = layers.LSTM(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_seq,
            name=f"{name_prefix}_{i}",
        )(x)

        # optional batch normalization
        if trial_batch_norm and trial.suggest_categorical(f"{name_prefix}_bn_{i}", [True, False]):
            x = layers.BatchNormalization(name=f"{name_prefix}_bn_{i}")(x)

        # apply residual connections if enabled
        x, state = apply_residual(x, trial, i, state, trial_skip_connections, name_prefix)

    return x
