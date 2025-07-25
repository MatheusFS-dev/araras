from araras.core import *

import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gc
import pandas as pd
import traceback
import os


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


def prune_model_by_config(
    trial: optuna.Trial,
    model: keras.Model,
    thresholds: Dict[str, float],
    *,
    bits_per_param: int = 32,
    batch_size: int = 1,
) -> None:
    """Prune the given trial if the model exceeds any configured limits.

    This helper computes several resource statistics for ``model`` and compares
    them against user provided ``thresholds``. If a threshold is surpassed the
    associated Optuna ``trial`` is pruned and a warning is logged.

    Supported keys for ``thresholds`` are:

    - ``"param"``: total number of parameters.
    - ``"model_size"``: model size in megabytes. MUST PROVIDE CORRECT BITS PER PARAMETER.
    - ``"memory_mb"``: estimated training memory in megabytes.
    - ``"flops"``: FLOPs for a single forward pass.

    Args:
        trial: The Optuna trial that may be pruned.
        model: Keras model to evaluate.
        thresholds: Mapping of pruning criteria to threshold values.
        bits_per_param: Bits used to store each parameter when calculating the
            model size. Defaults to ``32``.
        batch_size: Batch size used for memory estimation. Defaults to ``1``.

    Returns:
        None

    Raises:
        optuna.TrialPruned: If any threshold is exceeded.

    Notes:
        When multiple thresholds are provided the model will be pruned as soon
        as the first limit is violated.

    Warning:
        The memory estimation relies on :func:`estimate_training_memory` and is
        therefore only an approximation.
    """

    metrics = {
        "param": model.count_params(),
        "model_size": model.count_params() * bits_per_param / (8 * 1024 * 1024),
        "memory_mb": estimate_training_memory(model, batch_size=batch_size)
        / (1024 * 1024),
        "flops": get_flops(model, batch_size=1),
    }

    for key, threshold in thresholds.items():
        value = metrics.get(key)
        if value is None:
            continue
        if value > threshold:
            logger.warning(
                f"{YELLOW}Pruning trial {trial.number}: {key} {value:.2f} exceeds {threshold}{RESET}"
            )
            raise optuna.TrialPruned(f"Model exceeded {key} limit")


def plot_model_param_distribution(
    build_model_fn: Callable[[optuna.Trial], tf.keras.Model],
    bits_per_param: int,
    batch_size: int = 1,
    n_trials: int = 1000,
    fig_save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6),
    csv_path: Optional[str] = None,
    logs_dir: Optional[str] = None,
) -> None:
    """Sample random models, plot statistics and optionally save the results.

    This helper draws ``n_trials`` random models using ``build_model_fn`` and
    records their parameter counts, approximate model sizes and estimated
    training memory consumption. Histograms for each metric are displayed once
    sampling finishes. The TensorFlow session is cleared between trials to
    release GPU memory. Trials that raise ``tf.errors.ResourceExhaustedError``,
    ``tf.errors.InternalError``, ``tf.errors.UnavailableError`` or cuDNN scratch-space
    allocation errors are skipped. These types of errors usually mean Out of
    Memory (OOM) problems. The number of skipped trials is counted and printed
    at the end. When ``csv_path`` is provided, the collected statistics are
    saved to a CSV file including each trial's parameters and sorted in
    decreasing order of the estimated training memory. If ``logs_dir`` is set,
    parameters of failed trials along with the error traceback are saved to
    individual log files.

    Args:
        build_model_fn: Callable that receives an Optuna ``Trial`` and returns a
            compiled :class:`tf.keras.Model`.
        bits_per_param: Number of bits used to store each parameter.
        batch_size: Batch size used when estimating the training memory.
        n_trials: Total number of random trials to sample.
        fig_save_path: Optional path to save the figure. If ``None`` the figure
            is shown only.
        figsize: Figure size for the histograms.
        csv_path: Optional path to store trial results as CSV. The CSV includes
            the sampled parameters and is sorted by ``training_memory_mb`` in
            descending order.
        logs_dir: Directory where error logs are written. If ``None``, no logs
            are saved.

    Returns:
        None. The histograms are displayed using ``matplotlib``.

    Raises:
        None

    Notes:
        Clearing the Keras backend session between trials mitigates
        ``ResourceExhaustedError`` on GPUs with limited VRAM. When ``logs_dir``
        is provided, one log file per failed trial is created containing the
        trial parameters and traceback.

    Warning:
        Models that trigger ``ResourceExhaustedError`` are ignored in the final
        statistics.
    """
    config_plt("double-column")  # Configure matplotlib for double-column figures

    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")

    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    param_counts = []
    model_sizes_mb = []
    training_memory = []
    collected_params = []

    progress_iter = range(n_trials)
    if n_trials:
        progress_iter = white_track(
            progress_iter,
            description="Sampling models",
            total=n_trials,
        )

    # Counters for skipped trials
    oom_count = 0
    internal_error_count = 0
    unavailable_count = 0
    scratch_error_count = 0

    def _log_error(trial: optuna.Trial, err: BaseException) -> None:
        if not logs_dir:
            return
        log_file = os.path.join(logs_dir, f"trial_{trial.number}.log")
        with open(log_file, "w") as f:
            f.write(f"Params: {trial.params}\n")
            f.write(f"Error: {err}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    for _ in progress_iter:
        trial = study.ask()
        try:
            model = build_model_fn(trial)

            n_params = model.count_params()
            param_counts.append(n_params)

            size_mb = (n_params * bits_per_param) / (8 * 1024 * 1024)
            model_sizes_mb.append(size_mb)

            training_memory_mb = estimate_training_memory(model, batch_size=batch_size) / (1024 * 1024)
            training_memory.append(training_memory_mb)
            collected_params.append(trial.params)

            study.tell(trial, 0.0)
        except tf.errors.ResourceExhaustedError as e:
            oom_count += 1
            _log_error(trial, e)
            continue

        except tf.errors.InternalError as e:
            internal_error_count += 1
            _log_error(trial, e)
            continue

        except tf.errors.UnavailableError as e:
            unavailable_count += 1
            _log_error(trial, e)
            continue

        except tf.errors.UnknownError as e:
            # Skip cuDNN scratch‑space failures
            if "CUDNN failed to allocate the scratch space" in str(e):
                scratch_error_count += 1
                _log_error(trial, e)
                continue
            # re‑raise other UnknownError
            raise

        except Exception as e:
            print(f"Error during model sampling: {e}")
            traceback.print_exc()
            _log_error(trial, e)

        finally:
            if "model" in locals():
                del model
            tf.keras.backend.clear_session()
            gc.collect()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
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

    if fig_save_path:
        fig.savefig(fig_save_path, bbox_inches="tight", dpi=300)

    if csv_path:
        df = pd.DataFrame(
            {
                "param_count": param_counts,
                "model_size_mb": model_sizes_mb,
                "training_memory_mb": training_memory,
                "params": collected_params,
            }
        )
        df = df.sort_values("training_memory_mb", ascending=False)
        df.to_csv(csv_path, index=False)

    if oom_count:
        print(f"{RED}Skipped {oom_count} trial(s) due to ResourceExhaustedError.{RESET}")
        print(f"{RED}Skipped {internal_error_count} trial(s) due to InternalError.{RESET}")
        print(f"{RED}Skipped {unavailable_count} trial(s) due to UnavailableError.{RESET}")
        print(f"{RED}Skipped {scratch_error_count} trial(s) due to cuDNN scratch space error.{RESET}")


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
    
    Statistics include:
        - Number of parameters
        - Model size in bits
        - FLOPs (floating-point operations)
        - MACs (multiply-accumulate operations)
        - Model summary
        - Peak memory usage during inference
        - Inference time
        - Average power consumption
        - Average energy consumption

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
