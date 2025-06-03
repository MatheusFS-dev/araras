"""
This module provides a function to build a TCNN (Transposed Convolutional Neural Network) block with optional hyperparameter tuning and regularization.

Functions:
    - build_tcnn1d: Constructs a configurable 1D transposed convolutional layer with options for batch normalization, activation, and regularization.
    - build_tcnn2d: Constructs a configurable 2D transposed convolutional layer with options for batch normalization, activation, and regularization.
    - build_tcnn3d: Constructs a configurable 3D transposed convolutional layer with options for batch normalization, activation, and regularization.

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


def build_tcnn2d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    filters_range: Union[int, tuple[int, int]],
    filters_step: int,
    kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
    kernel_size_step: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: tuple[int, int] = (1, 1),
    dilation_rate: tuple[int, int] = (1, 1),
    use_bias: bool = False,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_batch_norm: bool = True,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    name_prefix: str = "tcnn2d",
) -> layers.Layer:
    """
    Builds a single 2D transposed convolution block with optional batch norm and activation.

    This function constructs a tunable Conv2DTranspose layer, optionally applies batch normalization,
    and concludes with a customizable activation function. Hyperparameter tuning is integrated via a `trial` object.

    Args:
        trial (Any): Hyperparameter tuning object, such as from Optuna.
        hparams (HParams): Hyperparameter manager providing activation and regularizer configurations.
        x (layers.Layer): Input layer/tensor to process.
        filters_range (Union[int, tuple[int, int]]): Fixed or tunable number of filters.
        filters_step (int): Step size for filter tuning.
        kernel_size_range (Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]):
            Fixed (height, width) or ranges ((h_min, h_max), (w_min, w_max)) for tuning.
        kernel_size_step (int): Step size for kernel size tuning.
        data_format (str): Format of the input data (e.g., "channels_last").
        padding (str): Type of padding to use in the convolution (e.g., "same" or "valid").
        strides (tuple[int, int]): Stride length for height and width.
        dilation_rate (tuple[int, int]): Dilation rate for height and width.
        use_bias (bool): Whether to include a bias term in the Conv2DTranspose layer.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for bias values.
        use_batch_norm (bool): Whether to apply batch normalization.
        trial_kernel_reg (bool): Whether to enable and tune kernel regularization.
        trial_bias_reg (bool): Whether to enable and tune bias regularization.
        trial_activity_reg (bool): Whether to enable and tune activity regularization.
        name_prefix (str): Prefix used for naming all internal layers.

    Returns:
        layers.Layer: Final output tensor after applying the Conv2DTranspose, optional batch norm, and activation.
    """

    # Determine number of filters for the Conv2DTranspose layer
    if isinstance(filters_range, int):
        filters = filters_range
    else:
        min_f, max_f = filters_range
        filters = trial.suggest_int(f"{name_prefix}_filters", min_f, max_f, step=filters_step)

    # Determine kernel size for the Conv2DTranspose layer
    if (
        isinstance(kernel_size_range, tuple)
        and isinstance(kernel_size_range[0], int)
        and isinstance(kernel_size_range[1], int)
    ):
        kernel_size = kernel_size_range  # Fixed height and width
    else:
        (h_min, h_max), (w_min, w_max) = kernel_size_range
        kernel_height = trial.suggest_int(f"{name_prefix}_kernel_height", h_min, h_max, step=kernel_size_step)
        kernel_width = trial.suggest_int(f"{name_prefix}_kernel_width", w_min, w_max, step=kernel_size_step)
        kernel_size = (kernel_height, kernel_width)

    # Get kernel regularizer if tuning is enabled
    kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None

    # Get bias regularizer if tuning is enabled
    bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None

    # Get activity regularizer if tuning is enabled
    act_reg = hparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    # Apply Conv2DTranspose layer
    x = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        activation=None,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        activity_regularizer=act_reg,
        name=name_prefix,
    )(x)

    # Optionally apply batch normalization
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    # Apply activation function as defined by hparams
    x = layers.Activation(
        hparams.get_activation(trial, f"{name_prefix}_act"),
        name=f"{name_prefix}_act",
    )(x)

    return x


def build_tcnn3d(
    trial: Any,
    hparams: HParams,
    x: layers.Layer,
    filters_range: Union[int, Tuple[int, int]],
    filters_step: int,
    kernel_size_range: Union[
        Tuple[int, int, int],
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    ],
    kernel_size_step: int,
    data_format: str = "channels_last",
    padding: str = "same",
    strides: Tuple[int, int, int] = (1, 1, 1),
    dilation_rate: Tuple[int, int, int] = (1, 1, 1),
    use_bias: bool = False,
    kernel_initializer: initializers.Initializer = initializers.GlorotUniform(),
    bias_initializer: initializers.Initializer = initializers.Zeros(),
    use_batch_norm: bool = True,
    trial_kernel_reg: bool = False,
    trial_bias_reg: bool = False,
    trial_activity_reg: bool = False,
    name_prefix: str = "tcnn3d",
) -> layers.Layer:
    """
    Builds a single 3D transposed convolution block with optional batch norm and activation.

    This function constructs a tunable Conv3DTranspose layer, optionally applies batch normalization,
    and concludes with a customizable activation function. Hyperparameter tuning is integrated via a `trial` object.

    Args:
        trial (Any): Hyperparameter tuning object, such as from Optuna.
        hparams (HParams): Hyperparameter manager providing activation and regularizer configurations.
        x (layers.Layer): Input layer/tensor to process.
        filters_range (Union[int, Tuple[int, int]]): Fixed or tunable number of filters.
        filters_step (int): Step size for filter tuning.
        kernel_size_range (Union[
            Tuple[int, int, int],
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        ]): Fixed (depth, height, width) or ranges ((d_min, d_max), (h_min, h_max), (w_min, w_max)) for tuning.
        kernel_size_step (int): Step size for kernel size tuning.
        data_format (str): Format of the input data (e.g., "channels_last").
        padding (str): Type of padding to use in the convolution (e.g., "same" or "valid").
        strides (Tuple[int, int, int]): Stride length for depth, height, and width.
        dilation_rate (Tuple[int, int, int]): Dilation rate for depth, height, and width.
        use_bias (bool): Whether to include a bias term in the Conv3DTranspose layer.
        kernel_initializer (initializers.Initializer): Initializer for kernel weights.
        bias_initializer (initializers.Initializer): Initializer for bias values.
        use_batch_norm (bool): Whether to apply batch normalization.
        trial_kernel_reg (bool): Whether to enable and tune kernel regularization.
        trial_bias_reg (bool): Whether to enable and tune bias regularization.
        trial_activity_reg (bool): Whether to enable and tune activity regularization.
        name_prefix (str): Prefix used for naming all internal layers.

    Returns:
        layers.Layer: Final output tensor after applying the Conv3DTranspose, optional batch norm, and activation.
    """

    # Determine number of filters for the Conv3DTranspose layer
    if isinstance(filters_range, int):
        filters = filters_range
    else:
        min_f, max_f = filters_range
        filters = trial.suggest_int(f"{name_prefix}_filters", min_f, max_f, step=filters_step)

    # Determine kernel size for the Conv3DTranspose layer
    if (
        isinstance(kernel_size_range, tuple)
        and isinstance(kernel_size_range[0], int)
        and isinstance(kernel_size_range[1], int)
        and isinstance(kernel_size_range[2], int)
    ):
        kernel_size = kernel_size_range  # Fixed depth, height, width
    else:
        (d_min, d_max), (h_min, h_max), (w_min, w_max) = kernel_size_range
        kernel_depth = trial.suggest_int(f"{name_prefix}_kernel_depth", d_min, d_max, step=kernel_size_step)
        kernel_height = trial.suggest_int(f"{name_prefix}_kernel_height", h_min, h_max, step=kernel_size_step)
        kernel_width = trial.suggest_int(f"{name_prefix}_kernel_width", w_min, w_max, step=kernel_size_step)
        kernel_size = (kernel_depth, kernel_height, kernel_width)

    # Get kernel regularizer if tuning is enabled
    kernel_reg = hparams.get_regularizer(trial, f"{name_prefix}_kernel_reg") if trial_kernel_reg else None

    # Get bias regularizer if tuning is enabled
    bias_reg = hparams.get_regularizer(trial, f"{name_prefix}_bias_reg") if trial_bias_reg else None

    # Get activity regularizer if tuning is enabled
    act_reg = hparams.get_regularizer(trial, f"{name_prefix}_act_reg") if trial_activity_reg else None

    # Apply Conv3DTranspose layer
    x = layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        activation=None,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        activity_regularizer=act_reg,
        name=name_prefix,
    )(x)

    # Optionally apply batch normalization
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

    # Apply activation function as defined by hparams
    x = layers.Activation(
        hparams.get_activation(trial, f"{name_prefix}_act"),
        name=f"{name_prefix}_act",
    )(x)

    return x
