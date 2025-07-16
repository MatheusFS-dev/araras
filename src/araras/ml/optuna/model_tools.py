"""
Last Edited: 14 July 2025
Description:
    Model management helpers for Optuna workflows.
"""
from araras.core import *

import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gc


from araras.ml.model.stats import get_flops, get_macs, get_memory_and_time, get_model_usage_stats
from araras.ml.model.utils import capture_model_summary

from araras.visualization.configs import config_plt

# ———————————————————————————————————————————————————————————————————————————— #
#                                   Utilities                                  #
# ———————————————————————————————————————————————————————————————————————————— #

def _get_model_trainable_params(model: keras.Model) -> int:
    """Get number of trainable parameters in the model."""
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])


def _get_precision_bytes(model: keras.Model) -> int:
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


def _get_optimizer_state_factor(model: keras.Model) -> int:
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


def _calculate_activation_memory(model: keras.Model, bytes_per_param: int) -> int:
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
            return int(_get_model_trainable_params(model) * bytes_per_param * 1.5)

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
        return int(_get_model_trainable_params(model) * bytes_per_param * 1.5)


def _get_framework_overhead() -> int:
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

# ———————————————————————————————————————————————————————————————————————————— #

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
    trainable_params = _get_model_trainable_params(model)
    bytes_per_param = _get_precision_bytes(model)
    state_factor = _get_optimizer_state_factor(model)

    # Calculate memory components
    param_memory = trainable_params * bytes_per_param * state_factor
    activation_memory_per_sample = _calculate_activation_memory(model, bytes_per_param)
    activation_memory = activation_memory_per_sample * batch_size
    framework_overhead = _get_framework_overhead()

    return param_memory + activation_memory + framework_overhead


def plot_model_param_distribution(
    build_model_fn: Callable[[optuna.Trial], tf.keras.Model],
    bits_per_param: int,
    batch_size: int = 1,
    n_trials: int = 1000,
    save_path: Optional[str] = None,
) -> None:
    """Sample random models and plot parameter and size histograms.

    This helper draws ``n_trials`` random models using ``build_model_fn`` and
    records their parameter counts, approximate model sizes and estimated
    training memory consumption. Histograms for each metric are displayed once
    sampling finishes. The TensorFlow session is cleared between trials to
    release GPU memory. Trials that raise ``tf.errors.ResourceExhaustedError``
    are skipped and the total number of skipped trials is printed at the end.

    Args:
        build_model_fn: Callable that receives an Optuna ``Trial`` and returns a
            compiled :class:`tf.keras.Model`.
        bits_per_param: Number of bits used to store each parameter.
        batch_size: Batch size used when estimating the training memory.
        n_trials: Total number of random trials to sample.
        save_path: Optional path to save the figure. If None, the figure is shown only.

    Returns:
        None. The histograms are displayed using ``matplotlib``.

    Raises:
        None

    Notes:
        Clearing the Keras backend session between trials mitigates
        ``ResourceExhaustedError`` on GPUs with limited VRAM.

    Warning:
        Models that trigger ``ResourceExhaustedError`` are ignored in the final
        statistics.
    """
    config_plt("double-column")  # Configure matplotlib for double-column figures

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
    oom_count = 0
    for _ in progress_iter:
        trial = study.ask()
        try:
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
        except tf.errors.ResourceExhaustedError:
            oom_count += 1
            continue
        finally:
            if "model" in locals():
                del model
            tf.keras.backend.clear_session()
            gc.collect()

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
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
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if oom_count:
        print(f"{RED}Skipped {oom_count} trial(s) due to ResourceExhaustedError.{RESET}")


def set_user_attr_model_stats(
    trial: optuna.Trial,
    model: tf.keras.Model,
    bits_per_param: int,
    batch_size: int,
    n_trials: int = 10000,
    device: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Extract and return model statistics from the given Optuna trial.

    Args:
        trial (optuna.Trial): The Optuna trial object
        model (tf.keras.Model): The Keras model to analyze.
        policy (tf.keras.DTypePolicy): The precision policy used for the model.
        batch_size (int): The batch size to simulate for input.
        n_trials (int): Number of trials for power and energy measurement.
        device (int): GPU index to run the model on. Use ``-1`` for CPU.
        verbose (bool): If True, print detailed information.

    Returns:
        Dict[str, float]: A dictionary containing model statistics
    """
    params = model.count_params()
    peak_mem_usage, inference_time = get_memory_and_time(
        model, batch_size=batch_size, device=device, verbose=verbose
    )
    _, avg_power, avg_energy = get_model_usage_stats(model, device=device, n_trials=n_trials, verbose=verbose)

    trial.set_user_attr("num_params", params)
    trial.set_user_attr("model_size", params * bits_per_param)
    trial.set_user_attr("flops", get_flops(model))
    trial.set_user_attr("macs", get_macs(model))
    trial.set_user_attr("model_summary", capture_model_summary(model))
    trial.set_user_attr("peak_memory_usage", peak_mem_usage)
    trial.set_user_attr("inference_time", inference_time)
    trial.set_user_attr("avg_power", avg_power)
    trial.set_user_attr("avg_energy", avg_energy)

    return {
        "num_params": params,
        "model_size": params * bits_per_param,
        "flops": get_flops(model),
        "macs": get_macs(model),
        "model_summary": capture_model_summary(model),
        "peak_memory_usage": peak_mem_usage,
        "inference_time": inference_time,
        "avg_power": avg_power,
        "avg_energy": avg_energy,
    }
