"""
Last Edited: 14 July 2025
Description:
    This script demonstrates a detailed 
    header format with additional metadata.
"""
from araras.core import *

import optuna
import tensorflow as tf
from tensorflow.keras import layers
from araras.ml.model.hyperparams import KParams


def build_squeeze_excite_1d(
    x: tf.keras.layers.Layer,
    trial: optuna.Trial,
    kparams: KParams,
    ratio_choices: List[int],
    name_prefix: str = "se_block",
) -> tf.keras.layers.Layer:
    """Apply a Squeeze-and-Excitation (SE) block 1D with Optuna-tuned hyperparameters.
    Based on the paper: https://arxiv.org/pdf/1709.01507

    Args:
        x: Input 3D tensor (batch, length, channels).
        trial: Optuna Trial object for suggesting hyperparameters.
        kparams: KParams object containing hyperparameter choices.
        ratio_choices: List of integers representing reduction ratios for SE block.
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
    act_reduce = kparams.get_activation(trial, f"{name_prefix}_se_act_reduce")
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
