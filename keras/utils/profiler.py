"""
This module provides utilities to calculate the number of floating-point operations (FLOPs)
and Multiply-Accumulate operations (MACs) for a given Keras model.

Functions:
    - get_flops: Calculates the total number of FLOPs for a single forward pass of the model.
    - get_macs: Estimates the number of MACs for a single forward pass of the model.
    
Example usage:
    flops = get_flops(model, batch_size=32)
    macs = get_macs(model, batch_size=32)
"""

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def get_flops(model: tf.keras.Model, batch_size: int = 1) -> int:
    """
    Calculates the total number of floating-point operations (FLOPs) needed
    to perform a single forward pass of the given Keras model.

    Flow:
        model -> input_shape -> TensorSpec -> tf.function -> concrete function
        -> graph -> profile(graph) -> total_float_ops -> return

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        batch_size (int, optional): The batch size to simulate for input. Defaults to 1.

    Returns:
        int: The total number of floating-point operations (FLOPs) for one forward pass.

    Raises:
        ValueError: If the model does not have a valid input shape.
    """

    # Retrieve the input shape of the model
    ishape = model.input_shape

    # Validate input shape; ensure it has more than one dimension (batch + data)
    if not ishape or len(ishape) < 2:
        raise ValueError("Model must have a valid input_shape")

    # Exclude the batch size from the input shape
    shape = tuple(ishape)[1:]

    # Display the inferred input shape (excluding batch dimension)
    print("Inferred input shape (excl. batch):", shape)

    # Define a TensorFlow TensorSpec with the given batch size and input shape
    spec = tf.TensorSpec([batch_size, *shape], dtype=model.inputs[0].dtype)

    # Wrap the model's call method in a tf.function for graph tracing
    forward_fn = tf.function(model.call, input_signature=[spec])

    # Obtain the concrete function (compiled graph) from the tf.function
    concrete = forward_fn.get_concrete_function()

    # Extract the computational graph from the concrete function
    graph = concrete.graph

    # Set profiling options to count floating-point operations
    opts = ProfileOptionBuilder.float_operation()

    # Profile the graph to compute FLOPs
    graph_info = profile(graph, options=opts)

    # Return the total number of floating-point operations
    return graph_info.total_float_ops


def get_macs(model: tf.keras.Model, batch_size: int = 1) -> int:
    """
    Estimates the number of Multiply-Accumulate operations (MACs) required
    for a single forward pass of the model. Assumes 1 MAC = 2 FLOPs.

    Flow:
        model -> input_shape -> TensorSpec -> tf.function -> concrete function
        -> graph -> profile(graph) -> total_float_ops // 2 -> return

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        batch_size (int, optional): The batch size to simulate for input. Defaults to 1.

    Returns:
        int: The estimated number of MACs for one forward pass.

    Raises:
        ValueError: If the model does not have a valid input shape.
    """

    # Retrieve the input shape of the model
    ishape = model.input_shape

    # Validate input shape; ensure it has more than one dimension (batch + data)
    if not ishape or len(ishape) < 2:
        raise ValueError("Model must have a valid input_shape")

    # Exclude the batch size from the input shape
    shape = tuple(ishape)[1:]

    # Display the inferred input shape (excluding batch dimension)
    print("Inferred input shape (excl. batch):", shape)

    # Define a TensorFlow TensorSpec with the given batch size and input shape
    spec = tf.TensorSpec([batch_size, *shape], dtype=model.inputs[0].dtype)

    # Wrap the model's call method in a tf.function for graph tracing
    forward_fn = tf.function(model.call, input_signature=[spec])

    # Obtain the concrete function (compiled graph) from the tf.function
    concrete = forward_fn.get_concrete_function()

    # Extract the computational graph from the concrete function
    graph = concrete.graph

    # Set profiling options to count floating-point operations
    opts = ProfileOptionBuilder.float_operation()

    # Profile the graph to compute FLOPs
    graph_info = profile(graph, options=opts)

    # Compute MACs by assuming 1 MAC = 2 FLOPs
    flops = graph_info.total_float_ops
    return flops // 2
