"""
This module provides a function to build a TCNN (Transposed Convolutional Neural Network) block with optional hyperparameter tuning and regularization.

Functions:
    - build_tcnn1d: Constructs a configurable 1D transposed convolutional layer with options for batch normalization, activation, and regularization.
    
Usage example:
    from araras.keras.builders.tcnn import build_tcnn1d
    from araras.keras.hparams import HParams
    import tensorflow as tf
    hparams = HParams()
    x = tf.keras.Input(shape=(128, 64))  # Example input shape
    tcnn_layer = build_tcnn1d(
        trial=trial,
        hparams=hparams,
        x=x,
        filters_range=(32, 128),
        filters_step=10,
        kernel_size_range=(3, 7),
        kernel_size_step=1,
        use_batch_norm=True,
        trial_kernel_reg=True,
        trial_bias_reg=True,
        trial_activity_reg=True
    )
"""

from typing import *
from tensorflow.keras import layers, initializers
from araras.keras.hparams import HParams


def build_tcnn1d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    filters_range: Union[int, tuple[int, int]],
    filters_step: int,
    kernel_size_range: Union[int, tuple[int, int]],
    kernel_size_step: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: int = 1,
    dilation_rate: int = 1,
    use_bias: bool = False,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_batch_norm: bool = True,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    name_prefix: str = "tcnn1d",
) -> layers.Layer:
    """
    Builds a single 1D transposed convolution block with optional batch norm and activation.

    This function constructs a tunable Conv1DTranspose layer, optionally applies batch normalization,
    and concludes with a customizable activation function. Hyperparameter tuning is integrated via a `trial` object.

    Args:
        trial (Any): Hyperparameter tuning object, such as from Optuna.
        hparams (HParams): Hyperparameter manager providing activation and regularizer configurations.
        x (layers.Layer): Input layer/tensor to process.
        filters_range (Union[int, tuple[int, int]]): Fixed or tunable number of filters.
        filters_step (int): Step size for filter tuning.
        kernel_size_range (Union[int, tuple[int, int]]): Fixed or tunable kernel size.
        kernel_size_step (int): Step size for kernel size tuning.
        data_format (str): Format of the input data (e.g., "channels_last").
        padding (str): Type of padding to use in the convolution (e.g., "same" or "valid").
        strides (int): Stride length of the convolution.
        dilation_rate (int): Dilation rate for dilated convolution.
        use_bias (bool): Whether to include a bias term in the Conv1DTranspose layer.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for bias values.
        use_batch_norm (bool): Whether to apply batch normalization.
        trial_kernel_reg (bool): Whether to enable and tune kernel regularization.
        trial_bias_reg (bool): Whether to enable and tune bias regularization.
        trial_activity_reg (bool): Whether to enable and tune activity regularization.
        name_prefix (str): Prefix used for naming all internal layers.

    Returns:
        layers.Layer: Final output tensor after applying the Conv1DTranspose, optional batch norm, and activation.

    Raises:
        None
    """

    # Determine number of filters for the Conv1DTranspose layer
    if isinstance(filters_range, int):
        filters = filters_range  # Use fixed number of filters
    else:
        min_f, max_f = filters_range  # Extract min and max from range
        # Suggest number of filters using trial object within defined range
        filters = trial.suggest_int(f"{name_prefix}_filters", min_f, max_f, step=filters_step)

    # Determine kernel size for the Conv1DTranspose layer
    if isinstance(kernel_size_range, int):
        kernel_size = kernel_size_range  # Use fixed kernel size
    else:
        min_k, max_k = kernel_size_range  # Extract min and max from range
        # Suggest kernel size using trial object within defined range
        kernel_size = trial.suggest_int(f"{name_prefix}_kernel_size", min_k, max_k, step=kernel_size_step)

    # Get kernel regularizer if tuning is enabled
    kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None

    # Get bias regularizer if tuning is enabled
    bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None

    # Get activity regularizer if tuning is enabled
    act_reg = hparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    # Apply Conv1DTranspose layer
    x = layers.Conv1DTranspose(
        filters=filters,  # Number of output filters
        kernel_size=kernel_size,  # Width of the 1D convolution window
        activation=None,  # Activation applied separately below
        use_bias=use_bias,  # Whether to include a bias term
        padding=padding,  # Padding type for convolution output
        data_format=data_format,  # Format of the input data
        strides=strides,  # Step size of the convolution
        dilation_rate=dilation_rate,  # Dilation rate for dilated convolution
        kernel_initializer=kernel_initializer,  # Initializer for kernel weights
        bias_initializer=bias_initializer,  # Initializer for bias values
        kernel_regularizer=kernel_reg,  # Optional kernel regularizer
        bias_regularizer=bias_reg,  # Optional bias regularizer
        activity_regularizer=act_reg,  # Optional activity regularizer
        name=name_prefix,  # Name assigned to this layer
    )(x)

    # Optionally apply batch normalization
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)  # Normalize outputs to stabilize learning

    # Apply activation function as defined by hparams
    x = layers.Activation(
        hparams.get_activation(trial, f"{name_prefix}_act"),  # Retrieve activation function from hparams
        name=f"{name_prefix}_act",
    )(x)

    return x  # Return the final output tensor after all transformations
