"""
This module provides a function to build a Deep Neural Network (DNN) block with optional hyperparameter tuning and regularization.

Functions:
    - build_dnn: Constructs a configurable DNN layer with options for dropout, batch normalization, and regularization.
    
Usage example:
    from araras.keras.builders.dnn import build_dnn
    from araras.keras.hparams import HParams
    import tensorflow as tf
    hparams = HParams()
    x = tf.keras.Input(shape=(128, 64))  # Example input shape
    dnn_layer = build_dnn(
        trial=trial,
        hparams=hparams,
        x=x,
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


def build_dnn(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    units_range: Union[int, tuple[int, int]],
    dropout_rate_range: Union[float, tuple[float, float]],
    units_step: int = 10,
    dropout_rate_step: float = 0.1,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_bias: bool = True,
    use_batch_norm: bool = False,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    name_prefix: str = "dnn",
) -> layers.Layer:
    """
    Builds a single dense neural network (DNN) block with optional regularization, batch normalization, and dropout.

    This function constructs a configurable DNN layer consisting of a Dense layer followed by optional
    batch normalization, a user-specified activation function, and dropout. It supports hyperparameter
    tuning via the `trial` object.

    Args:
        trial (Any): Hyperparameter tuning trial object, e.g., from Optuna.
        hparams (HParams): Custom hyperparameter handler that provides regularizers and activations.
        x (layers.Layer): Input tensor or layer to build on.
        units_range (Union[int, tuple[int, int]]): Either a fixed unit count or a range for tuning.
        units_step (int): Step size for unit range tuning.
        dropout_rate_range (Union[float, tuple[float, float]]): Either a fixed dropout rate or a range.
        dropout_rate_step (float): Step size for dropout rate tuning.
        kernel_initializer (initializers.Initializer): Initializer for Dense layer weights.
        bias_initializer (initializers.Initializer): Initializer for Dense layer biases.
        use_bias (bool): Whether to include a bias term in the Dense layer.
        use_batch_norm (bool): Whether to include a batch normalization layer.
        trial_kernel_reg (bool): Whether to tune and apply a kernel regularizer.
        trial_bias_reg (bool): Whether to tune and apply a bias regularizer.
        trial_activity_reg (bool): Whether to tune and apply an activity regularizer.
        name_prefix (str): Prefix to use for naming the layers.

    Returns:
        layers.Layer: Output tensor after applying the DNN block.

    Raises:
        None
    """

    # Determine the number of units for the Dense layer
    if isinstance(units_range, int):
        units = units_range  # Use the fixed number of units if provided
    else:
        min_u, max_u = units_range  # Extract bounds for units
        # Suggest integer number of units within range using the trial object
        units = trial.suggest_int(f"{name_prefix}_units", min_u, max_u, step=units_step)

    # Determine the dropout rate
    if isinstance(dropout_rate_range, float):
        dropout_rate = dropout_rate_range  # Use the fixed dropout rate if provided
    else:
        min_d, max_d = dropout_rate_range  # Extract bounds for dropout rate
        # Suggest float dropout rate within range using the trial object
        dropout_rate = trial.suggest_float(
            f"{name_prefix}_dropout", min_d, max_d, step=dropout_rate_step
        )

    # Configure kernel regularizer if requested
    kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None

    # Configure bias regularizer if requested
    bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None

    # Configure activity regularizer if requested
    act_reg = hparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    # Add a Dense layer without activation
    x = layers.Dense(
        units=units,  # Number of units as determined above
        activation=None,  # Activation applied separately below
        use_bias=use_bias,  # Whether to include a bias term
        kernel_initializer=kernel_initializer,  # Initializer for the weights
        bias_initializer=bias_initializer,  # Initializer for the biases
        kernel_regularizer=kernel_reg,  # Optional kernel regularizer
        bias_regularizer=bias_reg,  # Optional bias regularizer
        activity_regularizer=act_reg,  # Optional activity regularizer
        name=name_prefix,  # Use provided prefix for naming
    )(x)

    # Optionally apply Batch Normalization
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)  # Normalize batch with moving statistics

    # Apply activation function as determined by hparams
    x = layers.Activation(
        hparams.get_activation(trial, f"{name_prefix}_act"),  # Retrieve activation from hparams
        name=f"{name_prefix}_act",
    )(x)

    # Apply dropout for regularization
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)  # Randomly zero some elements

    return x  # Return the final output layer after all transformations
