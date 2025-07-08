"""
This module provides utility functions for penalizing loss values based on
model complexity metrics such as FLOPs (Floating Point Operations) and
the number of trainable parameters. These functions are useful for
regularizing models to encourage simpler architectures.

Functions:
    - compute_flops_penalized_loss: Penalizes loss based on model FLOPs.
    - compute_params_penalized_loss: Penalizes loss based on model parameters.

Example usage:
    new_loss = compute_flops_penalized_loss(val_loss, model, 1e-9, "add")
    new_loss = compute_params_penalized_loss(val_loss, model, 1e-8, "subtract")
"""

import tensorflow as tf
from araras.keras.utils.profiler import get_flops
from typing import Literal, Sequence, Union


def compute_flops_penalized_loss(
    loss: Union[float, Sequence[float]],
    model: tf.keras.Model,
    flops_penalty_factor: float = 1e-10,
    operation: Literal["add", "subtract"] = "subtract",
) -> Union[float, Sequence[float]]:
    # Validate operation mode
    if operation not in ("add", "subtract"):
        raise ValueError("`operation` must be either 'add' or 'subtract'.")

    total_flops = get_flops(model)

    # Compute penalty
    penalty = flops_penalty_factor * total_flops

    # Apply penalty to single value or list
    if isinstance(loss, (list, tuple)):
        return [
            l + penalty if operation == "add" else l - penalty
            for l in loss
        ]
    return loss + penalty if operation == "add" else loss - penalty


def compute_params_penalized_loss(
    loss: Union[float, Sequence[float]],
    model: tf.keras.Model,
    params_penalty_factor: float = 1e-9,
    operation: Literal["add", "subtract"] = "subtract",
) -> Union[float, Sequence[float]]:
    # Validate operation type
    if operation not in ("add", "subtract"):
        raise ValueError("`operation` must be either 'add' or 'subtract'.")

    # Count total trainable + non-trainable parameters
    total_params = model.count_params()

    # Compute penalty
    penalty = params_penalty_factor * total_params

    # Apply penalty to single value or list
    if isinstance(loss, (list, tuple)):
        return [
            l + penalty if operation == "add" else l - penalty
            for l in loss
        ]
    return loss + penalty if operation == "add" else loss - penalty
