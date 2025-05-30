"""
This module provides a function to build a LSTM block with optional hyperparameter tuning and regularization.

Functions:
    - build_lstm: Constructs a configurable LSTM layer with options for dropout, batch normalization, and regularization.

Usage example:
    from araras.keras.builders.lstm import build_lstm
    from araras.keras.hparams import HParams
    import tensorflow as tf
    hparams = HParams()
    x = tf.keras.Input(shape=(128, 64))  # Example input shape
    lstm_layer = build_lstm(
        trial=trial,
        hparams=hparams,
        x=x,
        return_sequences=True, # Use False for last output only
        units_range=(64, 256),
        units_step=32,
        dropout_rate_range=(0.1, 0.5),
        dropout_rate_step=0.1,
        use_batch_norm=True
    )
"""


from typing import *
from tensorflow.keras import layers, initializers
from araras.keras.hparams import HParams


def build_lstm(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    return_sequences: bool,
    units_range: Union[int, tuple[int, int]],
    units_step: int,
    dropout_rate_range: Union[float, tuple[float, float]],
    dropout_rate_step: float = 0.1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_bias: bool = True,
    use_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    name_prefix: str = "lstm",
) -> layers.Layer:
    """
    Builds a single LSTM block with optional regularization, batch normalization, and dynamic dropout.

    This function creates a tunable LSTM layer with optional regularization and batch normalization,
    followed by a customizable activation layer. It supports hyperparameter optimization through a tuning trial.

    Args:
        trial (Any): Object used for suggesting hyperparameters, typically from a tuner like Optuna.
        hparams (HParams): Hyperparameter manager used to retrieve regularizers and activations.
        x (layers.Layer): Input tensor or layer.
        return_sequences (bool): Whether to return the full sequence of outputs or just the last output.
        units_range (Union[int, tuple[int, int]]): Fixed or tunable number of LSTM units.
        units_step (int): Step size for tuning LSTM units if a range is given.
        dropout_rate_range (Union[float, tuple[float, float]]): Fixed or tunable dropout rate.
        dropout_rate_step (float): Step size for tuning dropout rate.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for biases.
        use_bias (bool): Whether to include a bias term in the LSTM layer.
        use_batch_norm (bool): Whether to apply batch normalization after LSTM.
        trial_kernel_reg (bool): Whether to apply/tune a kernel regularizer.
        trial_bias_reg (bool): Whether to apply/tune a bias regularizer.
        trial_activity_reg (bool): Whether to apply/tune an activity regularizer.
        name_prefix (str): Prefix to use for naming the layers.

    Returns:
        layers.Layer: Output tensor after applying the LSTM block.

    Raises:
        None
    """

    # Determine the number of units for the LSTM layer
    if isinstance(units_range, int):
        units = units_range  # Use fixed number of units
    else:
        min_u, max_u = units_range  # Unpack range of units
        # Suggest integer number of units within the range using trial object
        units = trial.suggest_int(f"{name_prefix}_units", min_u, max_u, step=units_step)

    # Determine the dropout rates
    if isinstance(dropout_rate_range, float):
        dropout_rate = dropout_rate_range  # Use fixed dropout rate
        recurrent_dropout_rate = dropout_rate_range  # Use same rate for recurrent dropout
    else:
        min_d, max_d = dropout_rate_range  # Unpack dropout rate range
        # Suggest float dropout rate for standard dropout
        dropout_rate = trial.suggest_float(
            f"{name_prefix}_dropout", min_d, max_d, step=dropout_rate_step
        )
        # Suggest float dropout rate for recurrent dropout
        recurrent_dropout_rate = trial.suggest_float(
            f"{name_prefix}_recurrent_dropout", min_d, max_d, step=dropout_rate_step
        )

    # Get kernel regularizer if enabled
    kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None

    # Get bias regularizer if enabled
    bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None

    # Get activity regularizer if enabled
    act_reg = hparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    # Apply LSTM layer without activation function
    x = layers.LSTM(
        units=units,  # Number of hidden units in the LSTM cell
        activation=None,  # Activation applied separately below
        use_bias=use_bias,  # Whether to use a bias term in the LSTM
        kernel_initializer=kernel_initializer,  # Weight initializer
        bias_initializer=bias_initializer,  # Bias initializer
        kernel_regularizer=kernel_reg,  # Optional regularizer for kernel
        bias_regularizer=bias_reg,  # Optional regularizer for bias
        activity_regularizer=act_reg,  # Optional regularizer for activity
        dropout=dropout_rate,  # Dropout rate for input connections
        recurrent_dropout=recurrent_dropout_rate,  # Dropout rate for recurrent state
        return_sequences=return_sequences,  # Return only last output in output sequence
        name=name_prefix,  # Name prefix for the LSTM layer
    )(x)

    # Optionally apply Batch Normalization
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)  # Normalize the activations

    # Apply activation function retrieved from hparams
    x = layers.Activation(
        hparams.get_activation(trial, f"{name_prefix}_act"),  # Get activation function from hparams
        name=f"{name_prefix}_act",
    )(x)

    return x  # Return final output tensor after LSTM, batch norm (optional), and activation
