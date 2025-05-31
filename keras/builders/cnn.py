"""
This module provides a function to build convolutional neural network (CNN) blocks.

Functions:
    - build_cnn1d: Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv1d: Simulates a Dense layer using a Conv1D layer with specific configurations.

Usage example:
    from araras.keras.builders.cnn import build_cnn1d
    from araras.keras.hparams import HParams
    import tensorflow as tf
    hparams = HParams()
    x = tf.keras.Input(shape=(128, 64))  # Example input shape
    cnn_layer = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=x,
        filters_range=(32, 128),
        kernel_size_range=(3, 7),
        use_batch_norm=True
    )
"""

from typing import *
from tensorflow.keras import layers, initializers
from araras.keras.hparams import HParams


def build_cnn1d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    filters_range: Union[int, tuple[int, int]],
    kernel_size_range: Union[int, tuple[int, int]],
    filters_step: int = 10,
    kernel_size_step: int = 1,
    use_batch_norm: bool = True,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    strides: int = 1,
    dilation_rate: int = 1,
    groups: int = 1,
    use_bias: bool = False,
    padding: str = "same",
    data_format: str = "channels_last",
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    name_prefix: str = "cnn1d",
) -> layers.Layer:
    """
    Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.

    This function creates a Conv1D layer whose hyperparameters (filters, kernel size, regularizers, etc.)
    can be either statically defined or dynamically tuned through an Optuna trial. It optionally applies
    batch normalization and a user-defined activation function.

    Args:
        trial (Any): An Optuna trial object used for hyperparameter optimization.
        hparams (HParams): A utility object to provide regularizers and activations.
        x (layers.Layer): The input Keras layer.
        filters_range (Union[int, tuple[int, int]]): Number of filters or a range for tuning.
        kernel_size_range (Union[int, tuple[int, int]]): Kernel size or a range for tuning.
        filters_step (int): Step size for filter tuning.
        kernel_size_step (int): Step size for kernel size tuning.
        use_batch_norm (bool): Whether to include batch normalization.
        trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
        trial_bias_reg (bool): Whether to tune and apply bias regularization.
        trial_activity_reg (bool): Whether to tune and apply activity regularization.
        strides (int): Stride size for the convolution.
        dilation_rate (int): Dilation rate for convolution.
        groups (int): Number of filter groups.
        use_bias (bool): Whether to use a bias term in the convolution. If using batch norm, this can be set to False.
        padding (str): Padding method ('valid' or 'same').
        data_format (str): Data format, either 'channels_last' or 'channels_first'.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for bias.
        name_prefix (str): Prefix for layer names.

    Returns:
        layers.Layer: The output Keras layer after applying convolution, optional batch norm, and activation.
    """

    # Determine number of filters: static value or tuned via Optuna.
    if isinstance(filters_range, int):
        filters = filters_range  # Use the provided static number of filters
    else:
        min_f, max_f = filters_range  # Unpack the minimum and maximum values for tuning
        filters = trial.suggest_int(
            f"{name_prefix}_filters", min_f, max_f, step=filters_step
        )  # Suggest a value from range

    # Determine kernel size: static value or tuned via Optuna.
    if isinstance(kernel_size_range, int):
        kernel_size = kernel_size_range  # Use the provided static kernel size
    else:
        min_k, max_k = kernel_size_range  # Unpack the range for kernel size
        kernel_size = trial.suggest_int(
            f"{name_prefix}_kernel_size", min_k, max_k, step=kernel_size_step
        )  # Suggest a value

    # Retrieve kernel regularizer if enabled
    kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None

    # Retrieve bias regularizer if enabled
    bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None

    # Retrieve activity regularizer if enabled
    act_reg = hparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    # Create the Conv1D layer with all specified and optional parameters, activation set to None
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation=None,  # Activation will be applied separately
        padding=padding,
        data_format=data_format,
        strides=strides,
        dilation_rate=dilation_rate,
        groups=groups,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        activity_regularizer=act_reg,
        name=name_prefix,
    )(x)

    # Optionally apply batch normalization after convolution and before activation
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    # Apply the activation function retrieved from hparams using the trial
    x = layers.Activation(
        hparams.get_activation(trial, f"{name_prefix}_act"),
        name=f"{name_prefix}_act",
    )(x)

    # Return the final transformed Keras layer
    return x


def build_dense_as_conv1d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    units: int,
    padding: str = "valid",
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    name_prefix: str = "dense_as_conv1d",
) -> layers.Layer:
    """
    Simulate a Dense layer using a Conv1D with kernel_size=1.

    This function builds a 1D convolutional layer that, when applied to a 3D input
    of shape (batch_size, length, features_in), produces an output of shape
    (batch_size, length, units).

    Note:
        If your goal is to emulate a classic Dense(units) on a flat vector of shape
        (batch_size, features_in), you must first reshape that vector to (batch_size, 1, features_in)
        and then apply this function. After Conv1D, you should call Flatten() to collapse
        back to (batch_size, units). Without reshaping, Conv1D will raise a shape mismatch
        on 2D inputs.
    
    Args:
        trial (Any): An Optuna trial object used for hyperparameter optimization.
        hparams (HParams): A utility object to provide regularizers and activations.
        x (layers.Layer): The input Keras layer, expected to be of shape (batch_size, length, features_in).
        units (int): The number of output units for the Dense layer.
        padding (str): Padding method ('valid' or 'same').
        trial_kernel_reg (bool): Whether to tune and apply kernel regularization.
        trial_bias_reg (bool): Whether to tune and apply bias regularization.
        trial_activity_reg (bool): Whether to tune and apply activity regularization.
        name_prefix (str): Prefix for layer names.

    Returns:
        layers.Layer: A Keras layer with output shape (batch_size, 1, units), equivalent to Dense(units).

    References:
        https://datascience.stackexchange.com/questions/12830 how-are-1x1-convolutions-the-same-as-a-fully-connected-layer

        https://www.educative.io/answers/are-1-x-1-convolutions-the-same-as-fully-connected-layers

        https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks
    """
    return build_cnn1d(
        trial,
        hparams,
        x,
        filters_range=units,
        kernel_size_range=1,
        strides=1,  
        padding=padding,
        use_bias=True,
        use_batch_norm=False,
        name_prefix=name_prefix,
        trial_kernel_reg=trial_kernel_reg,
        trial_bias_reg=trial_bias_reg,
        trial_activity_reg=trial_activity_reg,
    )
