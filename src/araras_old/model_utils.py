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
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from typing import Literal


def compute_flops_penalized_loss(
    loss: float,
    model: tf.keras.Model,
    flops_penalty_factor: float = 1e-10,
    operation: Literal["add", "subtract"] = "subtract",
) -> float:
    """
    Applies a penalty to the loss based on the number of FLOPs used by the model.

    Logic:
        -> Check that model input shape is valid
        -> Create a `tf.function` to profile the graph
        -> Use TensorFlow Profiler to estimate FLOPs
        -> Apply penalty using the given `flops_penalty_factor`
        -> Return penalized loss

    Args:
        loss (float): Original scalar loss (e.g., validation loss).
        model (tf.keras.Model): Keras model to be profiled.
        flops_penalty_factor (float): Scaling factor for FLOP penalty (default: 1e-10).
        operation (Literal["add", "subtract"]): Whether to add or subtract the penalty.

    Returns:
        float: Penalized loss value.

    Raises:
        ValueError: If input shape is missing or if operation is invalid.

    Example:
        new_loss = compute_flops_penalized_loss(val_loss, model, 1e-9, "add")
    """
    # Validate operation mode
    if operation not in ("add", "subtract"):
        raise ValueError("`operation` must be either 'add' or 'subtract'.")

    # Ensure model has a valid input shape
    input_shape = model.input_shape
    if not input_shape or len(input_shape) < 2:
        raise ValueError("Model `input_shape` must be defined and have at least 2 dimensions.")

    # Define a traced function for a single-sample inference
    forward_fn = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + tuple(input_shape[1:]), dtype=tf.float32)],
    )

    # Extract the computation graph
    graph = forward_fn.get_concrete_function().graph

    # Run profiler to get total floating-point operations (FLOPs)
    graph_info = profile(graph, options=ProfileOptionBuilder.float_operation())
    total_flops = graph_info.total_float_ops  # Integer count of FLOPs

    # Compute penalty and return adjusted loss
    penalty = flops_penalty_factor * total_flops
    return loss + penalty if operation == "add" else loss - penalty


def compute_params_penalized_loss(
    loss: float,
    model: tf.keras.Model,
    params_penalty_factor: float = 1e-9,
    operation: Literal["add", "subtract"] = "subtract",
) -> float:
    """
    Applies a penalty to the loss based on the number of trainable parameters in the model.

    Logic:
        -> Count model parameters
        -> Multiply by penalty factor
        -> Add or subtract penalty from original loss

    Args:
        loss (float): Original scalar loss (e.g., validation loss).
        model (tf.keras.Model): Keras model.
        params_penalty_factor (float): Scaling factor for parameter penalty (default: 1e-9).
        operation (Literal["add", "subtract"]): Whether to add or subtract the penalty.

    Returns:
        float: Adjusted loss after penalty.

    Raises:
        ValueError: If operation is not one of "add" or "subtract".

    Example:
        penalized = compute_params_penalized_loss(val_loss, model, 1e-8, "subtract")
    """
    # Validate operation type
    if operation not in ("add", "subtract"):
        raise ValueError("`operation` must be either 'add' or 'subtract'.")

    # Count total trainable + non-trainable parameters
    total_params = model.count_params()

    # Compute penalty and return adjusted loss
    penalty = params_penalty_factor * total_params
    return loss + penalty if operation == "add" else loss - penalty
