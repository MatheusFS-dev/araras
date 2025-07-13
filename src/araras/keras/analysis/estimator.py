"""
This module provides utilities for analyzing and visualizing the distribution of model parameters

Functions:
    - get_model_trainable_params: Get the number of trainable parameters in a Keras model.
    - get_precision_bytes: Determine bytes per parameter based on model's actual dtype.
    - get_optimizer_state_factor: Determine optimizer state factor from compiled model.
    - calculate_activation_memory: Calculate activation memory needed during forward/backward pass.
    - get_framework_overhead: Calculate framework overhead based on available GPU memory.
    - estimate_training_memory: Estimate total VRAM needed for training a Keras model in bytes
    - model_param_distribution: Sample random models and plot parameter, size and training memory necessity histograms.

Example:
    >>> from araras.keras.analysis.estimator import model_param_distribution
    >>> model_param_distribution(...)
"""

from araras.commons import *

import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from araras.utils import white_track

from araras.plot.configs import config_plt

config_plt("double-column")  # Configure matplotlib for double-column figures


def get_model_trainable_params(model: keras.Model) -> int:
    """Get number of trainable parameters in the model."""
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])


def get_precision_bytes(model: keras.Model) -> int:
    """Determine bytes per parameter based on model's actual dtype."""
    if hasattr(model, "dtype"):
        dtype = str(model.dtype)
    else:
        # Check first layer's dtype
        for layer in model.layers:
            if hasattr(layer, "dtype"):
                dtype = str(layer.dtype)
                break
        else:
            dtype = "float32"  # default

    if "float16" in dtype or "half" in dtype:
        return 2
    elif "float32" in dtype:
        return 4
    elif "float64" in dtype:
        return 8
    else:
        return 4  # default to fp32


def get_optimizer_state_factor(model: keras.Model) -> int:
    """Determine optimizer state factor from compiled model."""
    if not hasattr(model, "optimizer") or model.optimizer is None:
        return 3  # default to Adam-like

    optimizer_name = model.optimizer.__class__.__name__.lower()

    if "adam" in optimizer_name:
        return 3  # param + momentum + velocity
    elif "sgd" in optimizer_name:
        # Check if momentum is used
        if hasattr(model.optimizer, "momentum") and model.optimizer.momentum > 0:
            return 2  # param + momentum
        else:
            return 1  # param only
    elif "rmsprop" in optimizer_name:
        return 3  # param + accumulator + momentum
    elif "adagrad" in optimizer_name:
        return 2  # param + accumulator
    elif "adadelta" in optimizer_name:
        return 3  # param + accumulator + delta
    else:
        return 3  # default to Adam-like


def calculate_activation_memory(model: keras.Model, bytes_per_param: int) -> int:
    """Calculate activation memory needed during forward/backward pass."""
    try:
        # Get input shape from model
        if hasattr(model, "input_shape") and model.input_shape:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            # Use batch size of 1 to get per-sample memory, then multiply by actual batch
            if input_shape[0] is None:
                dummy_shape = (1,) + input_shape[1:]
            else:
                dummy_shape = input_shape
        else:
            # Fallback: estimate based on total parameters
            return int(get_model_trainable_params(model) * bytes_per_param * 1.5)

        # Create dummy input and trace through model
        dummy_input = tf.random.normal(dummy_shape)
        total_elements = 0
        x = dummy_input

        for layer in model.layers:
            try:
                x = layer(x)
                if hasattr(x, "shape"):
                    elements = tf.reduce_prod(
                        x.shape[1:]
                    ).numpy()  # Exclude batch dimension
                    total_elements += elements
            except:
                continue

        # Memory per sample * 2 (for gradients) * bytes_per_param
        return int(total_elements * 2 * bytes_per_param)

    except Exception:
        # Fallback calculation
        return int(get_model_trainable_params(model) * bytes_per_param * 1.5)


def get_framework_overhead() -> int:
    """Calculate framework overhead based on available GPU memory."""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # Scale overhead based on GPU memory (roughly 5-10% of total memory)
            gpu_details = tf.config.experimental.get_memory_info(gpus[0])
            total_gpu_memory = gpu_details["total"]
            return int(total_gpu_memory * 0.07)  # 7% of total GPU memory
        else:
            return 512 * 1024 * 1024  # 512 MB for CPU training
    except:
        return 1024 * 1024 * 1024  # 1 GB default


def estimate_training_memory(model: keras.Model, batch_size: int = 32) -> int:
    """
    Estimate total VRAM needed for training a Keras model in bytes.

    Args:
        model: Keras model object
        batch_size: Training batch size

    Returns:
        Total memory needed in bytes
    """
    # Get model characteristics
    trainable_params = get_model_trainable_params(model)
    bytes_per_param = get_precision_bytes(model)
    state_factor = get_optimizer_state_factor(model)

    # Calculate memory components
    param_memory = trainable_params * bytes_per_param * state_factor
    activation_memory_per_sample = calculate_activation_memory(model, bytes_per_param)
    activation_memory = activation_memory_per_sample * batch_size
    framework_overhead = get_framework_overhead()

    return param_memory + activation_memory + framework_overhead


def model_param_distribution(
    build_model_fn: Callable[[optuna.Trial], tf.keras.Model],
    bits_per_param: int,
    batch_size: int = 1,
    n_trials: int = 1000,
) -> None:
    """Sample random models and plot parameter and size histograms.

    Args:
        build_model_fn: Function that builds a Keras model given an Optuna
            ``Trial``.
        bits_per_param: Number of bits used to store each parameter.
        n_trials: Number of random trials to run.

    """

    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")

    param_counts = []
    model_sizes_mb = []
    training_memory = []

    progress_iter = range(n_trials)
    if n_trials:
        progress_iter = white_track(
            progress_iter,
            description="Sampling models",
            total=n_trials,
        )
    for _ in progress_iter:
        trial = study.ask()
        model = build_model_fn(trial)

        n_params = model.count_params()
        param_counts.append(n_params)

        size_mb = (n_params * bits_per_param) / (8 * 1024 * 1024)
        model_sizes_mb.append(size_mb)

        training_memory_mb = estimate_training_memory(model, batch_size=batch_size) / (
            1024 * 1024
        )
        training_memory.append(training_memory_mb)

        study.tell(trial, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].hist(param_counts, bins=100, color="black")
    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Parameter count distribution")

    axes[1].hist(model_sizes_mb, bins=100, color="black")
    axes[1].set_xlabel("Model size (MB)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Model size distribution")

    axes[2].hist(training_memory, bins=100, color="black")
    axes[2].set_xlabel("Training memory (MB)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Training memory distribution")

    plt.tight_layout()
    plt.show()
