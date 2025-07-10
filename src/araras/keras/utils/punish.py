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
from araras.commons import *

import tensorflow as tf
from araras.keras.utils.profiler import get_flops


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


# Function to facilitate the use of both penalties together
def punish_model(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    type: Literal["flops", "params", None] = None,
    flops_penalty_factor: float = 1e-10,
    params_penalty_factor: float = 1e-9,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Apply both FLOPs and parameter penalties to an objective.

    Args:
        target: Base objective value (scalar or list of scalars).
        model: Model whose complexity will be penalised.
        type: Type of penalty to apply, either "flops" or "params".
        flops_penalty_factor: Factor for FLOPs penalty.
        params_penalty_factor: Factor for parameters penalty.
        direction: Whether the objective should be minimised or maximised.

    Returns:
        The penalised objective value or list of values.
    """
    if type is None:
        # If no type is specified, return the target unchanged
        return target
    
    if type == "flops":
        target = punish_model_flops(
            target, model, flops_penalty_factor, direction
        )
    elif type == "params":
        target = punish_model_params(
            target, model, params_penalty_factor, direction
        )
    else:
        raise ValueError("`type` must be either 'flops', 'params' or None.")

    return target
