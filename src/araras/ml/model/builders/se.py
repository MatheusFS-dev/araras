from araras.core import *

import optuna
import tensorflow as tf
from tensorflow.keras import layers
from araras.ml.model.hyperparams import KParams


def build_squeeze_excite_1d(
    x: tf.keras.layers.Layer,
    trial: optuna.Trial,
    kparams: Optional[KParams],
    ratio_choices: List[int],
    act_reduce: Optional[Callable[..., Any]] = None,
    act_expand: Optional[Callable[..., Any]] = None,
    name_prefix: str = "se_block",
) -> tf.keras.layers.Layer:
    """Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters.
    Based on the paper: https://arxiv.org/pdf/1709.01507

    Args:
        x: Input 3D tensor (batch, length, channels).
        trial: Optuna Trial object for suggesting hyperparameters.
        kparams: KParams object containing hyperparameter choices. Can be ``None`` if
            both ``act_reduce`` and ``act_expand`` are provided.
        ratio_choices: List of integers representing reduction ratios for SE block.
        act_reduce: Optional activation for the reduction Dense layer. Overrides sampling
            from ``kparams`` when provided.
        act_expand: Optional activation for the expansion Dense layer. Overrides sampling
            from ``kparams`` when provided.
        name_prefix: Prefix for naming layers and trial parameters.

    Returns:
        A tensor the same shape as `x`, re-scaled by the SE attention weights.

    Raises:
        ValueError: If `x.shape[-1]` is None (undefined channel dimension).
    """
    # 1. Channel dimension
    channels = x.shape[-1]
    if channels is None:
        raise ValueError(f"Cannot infer channels for SE block with prefix {name_prefix}")

    # 2. Optuna suggestions using ratio_choices and KParams
    ratio = trial.suggest_categorical(f"{name_prefix}_se_ratio", ratio_choices)
    if act_reduce is None:
        if kparams is None:
            raise ValueError("kparams must be provided when act_reduce is None")
        act_reduce = kparams.get_activation(trial, f"{name_prefix}_se_act_reduce")
    if act_expand is None:
        if kparams is None:
            raise ValueError("kparams must be provided when act_expand is None")
        act_expand = kparams.get_activation(trial, f"{name_prefix}_se_act_expand")

    # 3. Squeeze
    se = layers.GlobalAveragePooling1D(name=f"{name_prefix}_se_squeeze")(x)
    # 4. Excitation: reduce → expand
    se = layers.Dense(
        units=channels // ratio,
        activation=act_reduce,
        name=f"{name_prefix}_se_reduce",
    )(se)
    se = layers.Dense(
        units=channels,
        activation=act_expand,
        name=f"{name_prefix}_se_expand",
    )(se)
    # 5. Reshape and scale
    se = layers.Reshape((1, channels), name=f"{name_prefix}_se_reshape")(se)
    return layers.Multiply(name=f"{name_prefix}_se_scale")([x, se])


def build_squeeze_excite_2d(
    x: tf.keras.layers.Layer,
    trial: optuna.Trial,
    kparams: Optional[KParams],
    ratio_choices: List[int],
    act_reduce: Optional[Callable[..., Any]] = None,
    act_expand: Optional[Callable[..., Any]] = None,
    name_prefix: str = "se_block_2d",
) -> tf.keras.layers.Layer:
    """
    Apply a 2D Squeeze-and-Excitation block.

    Args:
        x: Input tensor of shape (batch, height, width, channels).
        trial: Optuna Trial for hyperparameter suggestions.
        hparams: HParams object for activation choices. Can be ``None`` if both
            ``act_reduce`` and ``act_expand`` are provided.
        ratio_choices: List of reduction ratios to try.
        act_reduce: Optional activation for the reduction Dense layer. Overrides
            sampling from ``kparams`` when provided.
        act_expand: Optional activation for the expansion Dense layer. Overrides
            sampling from ``kparams`` when provided.
        name_prefix: Prefix for layer names.

    Returns:
        Tensor of same shape as `x`, re-weighted by channel attention.

    Raises:
        ValueError: if channel dimension is undefined.
    """
    # 1. Infer channels
    channels = x.shape[-1]
    if channels is None:
        raise ValueError(f"Cannot infer channels for SE block with prefix {name_prefix}")

    # 2. Hyperparameter suggestions
    ratio = trial.suggest_categorical(f"{name_prefix}_se_ratio", ratio_choices)
    if act_reduce is None:
        if kparams is None:
            raise ValueError("kparams must be provided when act_reduce is None")
        act_reduce = kparams.get_activation(trial, f"{name_prefix}_se_act_reduce")
    if act_expand is None:
        if kparams is None:
            raise ValueError("kparams must be provided when act_expand is None")
        act_expand = kparams.get_activation(trial, f"{name_prefix}_se_act_expand")

    # 3. Squeeze: global average over spatial dims
    se = layers.GlobalAveragePooling2D(name=f"{name_prefix}_se_squeeze")(x)

    # 4. Excitation: bottleneck dense layers
    se = layers.Dense(
        units=channels // ratio,
        activation=act_reduce,
        name=f"{name_prefix}_se_reduce",
    )(se)
    se = layers.Dense(
        units=channels,
        activation=act_expand,
        name=f"{name_prefix}_se_expand",
    )(se)

    # 5. Reshape for channel scaling
    se = layers.Reshape((1, 1, channels), name=f"{name_prefix}_se_reshape")(se)

    # 6. Scale input by SE weights
    return layers.Multiply(name=f"{name_prefix}_se_scale")([x, se])
