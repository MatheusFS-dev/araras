"""Utilities for penalizing objective values based on model complexity.

This module contains helpers that adjust an objective value (usually a loss
or metric) according to the computational cost of a Keras model.  Two types of
penalties are provided: one based on the number of FLOPs required for a single
forward pass and another based on the number of model parameters.  These are
useful for discouraging overly complex models during optimisation.

Example:
    >>> adjusted = punish_model_flops(val_loss, model, 1e-9, "minimize")
    >>> adjusted = punish_model_params(val_loss, model, 1e-8, "maximize")
"""

import tensorflow as tf
from araras.keras.utils.profiler import get_flops
from typing import Literal, Sequence, Union


def punish_model_flops(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-10,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's FLOPs.

    Args:
        target: Base objective value (scalar or list of scalars).
        model: Model whose FLOPs will be used for the penalty.
        penalty_factor: Multiplicative factor applied to the FLOPs count.
        direction: Whether the objective should be minimised or maximised.

    Returns:
        The penalised objective value or list of values.
    """

    if direction not in ("minimize", "maximize"):
        raise ValueError("`direction` must be either 'minimize' or 'maximize'.")

    total_flops = get_flops(model)

    # Compute penalty
    penalty = penalty_factor * total_flops

    # Apply penalty to single value or list
    if isinstance(target, (list, tuple)):
        return [
            t + penalty if direction == "minimize" else t - penalty
            for t in target
        ]
    return target + penalty if direction == "minimize" else target - penalty


def punish_model_params(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-9,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's parameter count.

    Args:
        target: Base objective value (scalar or list of scalars).
        model: Model whose parameters will be used for the penalty.
        penalty_factor: Multiplicative factor applied to the parameter count.
        direction: Whether the objective should be minimised or maximised.

    Returns:
        The penalised objective value or list of values.
    """

    if direction not in ("minimize", "maximize"):
        raise ValueError("`direction` must be either 'minimize' or 'maximize'.")

    # Count total trainable + non-trainable parameters
    total_params = model.count_params()

    # Compute penalty
    penalty = penalty_factor * total_params

    # Apply penalty to single value or list
    if isinstance(target, (list, tuple)):
        return [
            t + penalty if direction == "minimize" else t - penalty
            for t in target
        ]
    return target + penalty if direction == "minimize" else target - penalty
