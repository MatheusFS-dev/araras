from araras.core import *

from typing import Any, Dict, Iterable, Tuple, Union
from collections import Counter

import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gc
import pandas as pd
import numpy as np
import traceback
import os
import math

from araras.ml.model.stats import get_flops, get_macs, get_memory_and_time, get_model_usage_stats
from araras.ml.model.utils import capture_model_summary
from araras.utils.misc import format_number, format_bytes

from araras.visualization.configs import config_plt

from araras.ml.model.tools import save_model_plot

# ———————————————————————————————————————————————————————————————————————————— #
#                                   Utilities                                  #
# ———————————————————————————————————————————————————————————————————————————— #

def _get_model_trainable_params(model: keras.Model) -> int:
    """Get number of trainable parameters in the model."""
    return sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)


def _get_model_non_trainable_params(model: keras.Model) -> int:
    """Get number of non-trainable parameters in the model."""
    return sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)


def _resolve_dtype_bytes(dtype: Union[str, tf.dtypes.DType]) -> int:
    """Return byte-width for the provided TensorFlow dtype."""
    try:
        tf_dtype = tf.as_dtype(dtype)
        # ``size`` returns the number of bytes occupied by the dtype.
        size = tf_dtype.size
        if size is None:
            return 4
        return int(size)
    except TypeError:
        return 4


def _get_precision_bytes(model: keras.Model) -> Tuple[int, int]:
    """Return (variable_bytes, compute_bytes) for the model."""

    policy = getattr(model, "dtype_policy", None)
    variable_dtype = getattr(policy, "variable_dtype", None)
    compute_dtype = getattr(policy, "compute_dtype", None)

    if variable_dtype is None:
        # Fallback to the dtype of the first weight tensor.
        for weight in model.weights:
            variable_dtype = weight.dtype
            break

    if compute_dtype is None:
        if hasattr(model, "compute_dtype") and model.compute_dtype:
            compute_dtype = model.compute_dtype
        elif hasattr(model, "dtype") and model.dtype:
            compute_dtype = model.dtype
        else:
            for layer in model.layers:
                if hasattr(layer, "dtype") and layer.dtype:
                    compute_dtype = layer.dtype
                    break

    variable_bytes = _resolve_dtype_bytes(variable_dtype or "float32")
    compute_bytes = _resolve_dtype_bytes(compute_dtype or variable_dtype or "float32")

    return variable_bytes, compute_bytes


def _get_optimizer_slot_factor(model: keras.Model) -> int:
    """Return the number of optimizer slot tensors kept per parameter."""

    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        # Assume SGD without momentum plus gradient tensor handled elsewhere.
        return 0

    optimizer_name = optimizer.__class__.__name__.lower()

    if "adam" in optimizer_name or "nadam" in optimizer_name or "adamax" in optimizer_name:
        return 2  # first and second moments
    if "adafactor" in optimizer_name:
        # Adafactor keeps factored second moment states in practice (approx 2 slots)
        return 2
    if "rmsprop" in optimizer_name:
        # mean_square plus momentum (if enabled)
        has_momentum = getattr(optimizer, "momentum", 0) > 0
        return 2 if has_momentum else 1
    if "adagrad" in optimizer_name:
        return 1
    if "adadelta" in optimizer_name:
        return 2
    if "ftrl" in optimizer_name:
        # accumulators + linear slots
        return 2
    if "sgd" in optimizer_name:
        return 1 if getattr(optimizer, "momentum", 0) > 0 else 0

    # Default to two slots for unknown adaptive optimizers.
    return 2


def _iter_tensor_shapes(shapes: Union[tf.TensorShape, Iterable, None]) -> Iterable[tf.TensorShape]:
    """Yield TensorShape objects from possibly nested iterables."""

    if shapes is None:
        return

    if isinstance(shapes, tf.TensorShape):
        yield shapes
        return

    if isinstance(shapes, (tuple, list)):
        # Distinguish between a single shape tuple ``(None, 128)`` and a list of shapes.
        if not shapes:
            return

        if all(isinstance(dim, (int, type(None))) for dim in shapes):
            yield tf.TensorShape(shapes)
            return

        for item in shapes:
            yield from _iter_tensor_shapes(item)
        return

    # Fallback: best effort conversion.
    yield tf.TensorShape(shapes)


def _shape_num_elements(
    shape: Union[tf.TensorShape, Iterable, None], batch_size: int
) -> int:
    """Resolve the number of elements encoded by ``shape``.

    Args:
        shape (Union[tf.TensorShape, Iterable, None]): Tensor shape that may
            include ``None`` dimensions.
        batch_size (int): Batch dimension used to replace leading ``None``.

    Returns:
        int: Total element count after resolving undefined dimensions.
    """

    if shape is None:
        return 0

    tensor_shape = tf.TensorShape(shape)
    if tensor_shape.rank is None:
        return 0

    resolved_dims = []
    for index, dim in enumerate(tensor_shape):
        if dim is None:
            resolved_dims.append(batch_size if index == 0 else 1)
        else:
            resolved_dims.append(int(dim))

    if not resolved_dims:
        return 0

    return int(math.prod(resolved_dims))


def _tensor_num_bytes(
    tensor: Any,
    batch_size: int,
    dtype_bytes: int,
) -> int:
    """Compute the memory footprint of ``tensor`` using symbolic metadata.

    Args:
        tensor (Any): Keras tensor or placeholder exposing a ``shape`` attribute.
        batch_size (int): Batch dimension used for ``None`` resolution.
        dtype_bytes (int): Byte width of the tensor dtype.

    Returns:
        int: Size of the tensor in bytes or ``0`` if the shape is unknown.
    """

    shape = getattr(tensor, "shape", None)
    elements = _shape_num_elements(shape, batch_size)
    return elements * dtype_bytes


def _estimate_graph_activation_bytes(
    model: keras.Model,
    batch_size: int,
    activation_bytes: int,
) -> int:
    """Estimate the forward activation footprint using graph liveness analysis.

    Args:
        model (keras.Model): Graph model exposing ``_nodes_by_depth``.
        batch_size (int): Batch dimension used for symbolic tensors.
        activation_bytes (int): Byte width used for activation tensors.

    Returns:
        int: Peak number of bytes occupied by live forward activations.

    Raises:
        ValueError: If the model lacks graph metadata for static analysis.
    """

    nodes_by_depth = getattr(model, "_nodes_by_depth", None)
    if not nodes_by_depth:
        raise ValueError("Model does not expose graph nodes for static analysis.")

    ordered_nodes = []
    for depth in sorted(nodes_by_depth.keys()):
        ordered_nodes.extend(nodes_by_depth[depth])

    input_tensors_per_node = []
    consumer_counts: Counter[int] = Counter()

    for node in ordered_nodes:
        input_tensors = tf.nest.flatten(getattr(node, "input_tensors", ()))
        input_tensors_per_node.append(input_tensors)
        for tensor in input_tensors:
            consumer_counts[id(tensor)] += 1

    live_bytes = 0
    peak_bytes = 0
    tensor_sizes: Dict[int, int] = {}

    for node, input_tensors in zip(ordered_nodes, input_tensors_per_node):
        output_tensors = tf.nest.flatten(getattr(node, "output_tensors", ()))

        for tensor in output_tensors:
            tensor_id = id(tensor)
            size_bytes = _tensor_num_bytes(tensor, batch_size, activation_bytes)
            if size_bytes <= 0:
                continue
            tensor_sizes[tensor_id] = size_bytes
            live_bytes += size_bytes

        peak_bytes = max(peak_bytes, live_bytes)

        for tensor in input_tensors:
            tensor_id = id(tensor)
            if tensor_id not in tensor_sizes:
                continue

            consumer_counts[tensor_id] -= 1
            if consumer_counts[tensor_id] <= 0:
                live_bytes -= tensor_sizes.pop(tensor_id, 0)

        peak_bytes = max(peak_bytes, live_bytes)

    return int(peak_bytes)


def _estimate_peak_activation_bytes(
    model: keras.Model,
    batch_size: int,
    activation_bytes: int,
    *,
    activation_overhead_factor: float = 0.10,
) -> int:
    """Estimate activation residency including transient backward buffers.

    Args:
        model (keras.Model): Model to analyse for activation liveness.
        batch_size (int): Batch size assumed for symbolic tensors.
        activation_bytes (int): Byte width used for activation tensors.
        activation_overhead_factor (float): Fraction of forward bytes kept to
            approximate short-lived backward buffers.

    Returns:
        int: Estimated activation memory in bytes.
    """

    try:
        peak_forward = _estimate_graph_activation_bytes(
            model, batch_size, activation_bytes
        )
    except Exception:
        total_elements = 0
        for layer in model.layers:
            shapes = getattr(layer, "output_shape", None)
            if shapes is None and hasattr(layer, "output"):
                shapes = tf.keras.backend.int_shape(layer.output)

            for shape in _iter_tensor_shapes(shapes):
                total_elements += _shape_num_elements(shape, batch_size)

        peak_forward = total_elements * activation_bytes

    if peak_forward <= 0:
        return 0

    overhead = peak_forward * activation_overhead_factor
    return int(peak_forward + overhead)


def _estimate_device_overhead_bytes() -> int:
    """Estimate baseline runtime overhead for the active accelerator.

    Returns:
        int: Bytes reserved for device runtime, allocator, and workspace needs.
    """

    cpu_floor = 128 * 1024 * 1024  # 128 MB accounts for TF runtime caches.
    gpu_floor = 384 * 1024 * 1024  # 384 MB approximates CUDA context + allocator.
    gpu_fraction = 0.12
    gpu_cap_fraction = 0.25

    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            details = tf.config.experimental.get_device_details(gpus[0])
            memory_limit = (
                details.get("memory_limit") if isinstance(details, dict) else None
            )
            if memory_limit:
                baseline = int(memory_limit * gpu_fraction)
                cap = int(memory_limit * gpu_cap_fraction)
                overhead = max(gpu_floor, min(baseline, cap))
                return overhead
            return gpu_floor
    except Exception:
        pass

    return cpu_floor

# ———————————————————————————————————————————————————————————————————————————— #

def estimate_training_memory(model: keras.Model, batch_size: int = 32) -> int:
    """Estimate the VRAM footprint required to train ``model``.

    Args:
        model (keras.Model): Model to analyse. ``model`` must be built so that
            layer shapes are available. Functional/Sequential models receive a
            static liveness sweep; subclassed models fall back to a conservative
            approximation.
        batch_size (int): Mini-batch size used during training.

    Returns:
        int: Estimated memory requirement in bytes including tensors and runtime
        overhead.

    Notes:
        The estimation assumes that variable tensors and optimizer slots use the
        variable dtype while forward activations follow the compute dtype.
        Activation residency leverages a static liveness analysis for graph
        models and reverts to a parameter-proportional upper bound when graph
        structure is unavailable.

    Warnings:
        This function produces an analytical approximation. CUDA workspaces,
        allocator behaviour, and layer-specific buffers can change with driver
        versions or custom ops. Always validate against runtime telemetry when
        operating close to device limits.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Get model characteristics
    trainable_params = _get_model_trainable_params(model)
    non_trainable_params = _get_model_non_trainable_params(model)
    variable_bytes, compute_bytes = _get_precision_bytes(model)
    slot_factor = _get_optimizer_slot_factor(model)

    # Parameter tensors (weights)
    weight_bytes = trainable_params * variable_bytes
    non_trainable_bytes = non_trainable_params * variable_bytes

    # Gradients are materialised in the variable dtype when applied.
    gradient_bytes = trainable_params * variable_bytes

    # Optimizer slots (e.g., Adam moments, momentum buffers)
    optimizer_slot_bytes = trainable_params * variable_bytes * slot_factor

    # Peak forward activations retained by autodiff plus transient buffers.
    activation_memory = _estimate_peak_activation_bytes(
        model,
        batch_size,
        activation_bytes=compute_bytes,
    )

    base_memory = weight_bytes + non_trainable_bytes + gradient_bytes + optimizer_slot_bytes + activation_memory
    framework_overhead = _estimate_device_overhead_bytes()

    return int(base_memory + framework_overhead)


def prune_model_by_config(
    trial: optuna.Trial,
    model: keras.Model,
    thresholds: Dict[str, float],
    *,
    bytes_per_param: int = 8,
    batch_size: int = 1,
    verbose: bool = False,
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
        trial (optuna.Trial): The Optuna trial that may be pruned.
        model (keras.Model): Keras model to evaluate.
        thresholds (Dict[str, float]): Mapping of pruning criteria to threshold values.
        bytes_per_param (int): Bytes used to store each parameter when calculating the model size. Defaults to ``8``.
        batch_size (int): Batch size used for memory estimation. Defaults to ``1``.
        verbose (bool): If True, print stats for the model.

    Raises:
        optuna.TrialPruned: If any threshold is exceeded.

    Notes:
        When multiple thresholds are provided the model will be pruned as soon
        as the first limit is violated.

    Warnings:
        The memory estimation relies on :func:`estimate_training_memory` and is
        therefore only an approximation.
    """
    
    # Do nothing if user pass thresholds={}
    if not thresholds:
        return

    metrics = {
        "param": model.count_params(),
        "model_size": model.count_params() * bytes_per_param / (1024 * 1024),
        "memory_mb": estimate_training_memory(model, batch_size=batch_size)
        / (1024 * 1024),
        "flops": get_flops(model, batch_size=1),
    }
    
    if verbose:
        print(f"\nModel stats for trial {trial.number}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        print()

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
    bytes_per_param: int,
    batch_size: Union[int, Iterable[int]] = 1,
    n_trials: int = 1000,
    fig_save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6),
    csv_path: Optional[str] = None,
    logs_dir: Optional[str] = None,
    corr_csv_path: Optional[str] = None,
    plot_model_dir: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Sample random models, plot statistics and optionally save the results.

    The helper draws ``n_trials`` random models using ``build_model_fn`` and records
    their parameter counts, approximate model sizes and estimated training memory
    consumption. Histograms for each metric can be saved to disk and, optionally,
    displayed after sampling completes. The TensorFlow session is cleared between
    trials to release GPU memory. Trials that raise ``tf.errors.ResourceExhaustedError``,
    ``tf.errors.InternalError``, ``tf.errors.UnavailableError`` or cuDNN scratch-space
    allocation errors are skipped because they typically indicate Out of Memory
    (OOM) problems. The number of skipped trials is counted and printed at the end.
    When ``csv_path`` is provided, the collected statistics are saved to a CSV file,
    including each trial's parameters and sorted in decreasing order of the estimated
    training memory. If ``logs_dir`` is set, parameters of failed trials along with the
    error traceback are saved to individual log files.

    Args:
        build_model_fn (Callable[[optuna.Trial], tf.keras.Model]): Callable that receives an Optuna ``Trial`` and returns a compiled
            :class:`tf.keras.Model`.
        bytes_per_param (int): Number of bytes used to store each parameter.
        batch_size (Union[int, Iterable[int]]): Batch size used when estimating the training memory. Can be a single
            int or an iterable of ints; when multiple values are provided they are all evaluated.
        n_trials (int): Total number of random trials to sample.
        fig_save_path (Optional[str]): Optional path to save the figure. If ``None`` and ``show_plot`` is
            ``True``, the figure is only displayed.
        figsize (Tuple[int, int]): Figure size for the histograms.
        csv_path (Optional[str]): Optional path to store trial results as CSV. The CSV includes the sampled
            parameters and is sorted by ``training_memory_mb`` in descending order.
        logs_dir (Optional[str]): Directory where error logs are written. If ``None``, no logs are saved.
        corr_csv_path (Optional[str]): Optional path to store correlations between numeric hyperparameters and
            the model parameter count. If ``None`` the correlation analysis is skipped.
        plot_model_dir (Optional[str]): Directory where model plots are saved. If ``None``, no plots are saved.
            Each model is saved as a PNG file named ``model_{trial_number}.png``.
        show_plot (bool): Whether to display the histogram figure after sampling. Defaults to ``False``.

    Notes:
        Clearing the Keras backend session between trials mitigates
        ``ResourceExhaustedError`` on GPUs with limited VRAM. When ``logs_dir`` is provided,
        one log file per failed trial is created containing the trial parameters and traceback.
        Graphviz must be installed to plot models.

    Warnings:
        Models that trigger ``ResourceExhaustedError`` are ignored in the final statistics.
    """
    config_plt("double-column")  # Configure matplotlib for double-column figures

    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")

    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    if isinstance(batch_size, (int, np.integer)):
        batch_sizes = [int(batch_size)]
    elif isinstance(batch_size, (list, tuple)):
        if not batch_size:
            raise ValueError("batch_size iterable must contain at least one value.")
        normalized_batch_sizes = []
        for value in batch_size:
            if not isinstance(value, (int, np.integer)):
                raise TypeError("batch_size iterable must contain only integers.")
            normalized_batch_sizes.append(int(value))
        batch_sizes = list(dict.fromkeys(normalized_batch_sizes))
    else:
        raise TypeError("batch_size must be an int or a list/tuple of ints.")

    param_counts = []
    model_sizes_mb = []
    training_memory_map = {bs: [] for bs in batch_sizes}
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

            size_mb = (n_params * bytes_per_param) / (8 * 1024 * 1024)
            model_sizes_mb.append(size_mb)

            for batch in batch_sizes:
                training_memory_mb = (
                    estimate_training_memory(model, batch_size=batch) / (1024 * 1024)
                )
                training_memory_map[batch].append(training_memory_mb)
            collected_params.append(trial.params)

            if plot_model_dir:
                os.makedirs(plot_model_dir, exist_ok=True)
                model_path = os.path.join(plot_model_dir, f"model_{trial.number}.png")

                try:
                    save_model_plot(
                        model,
                        model_path,
                    )
                except Exception as e:
                    logger_error.error(f"{RED} Failed to plot model {trial.number}: {e} {RESET}")
                    traceback.print_exc()

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

    total_axes = 2 + len(batch_sizes)
    base_axes = 3
    if total_axes != base_axes:
        width_scale = total_axes / base_axes
        dynamic_figsize = (figsize[0] * width_scale, figsize[1])
    else:
        dynamic_figsize = figsize
    fig, axes = plt.subplots(1, total_axes, figsize=dynamic_figsize)
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]

    axes_list[0].hist(param_counts, bins=100, color="black")
    axes_list[0].set_xlabel("Number of parameters")
    axes_list[0].set_ylabel("Frequency")
    axes_list[0].set_title("Parameter count distribution")

    axes_list[1].hist(model_sizes_mb, bins=100, color="black")
    axes_list[1].set_xlabel("Model size (MB)")
    axes_list[1].set_ylabel("Frequency")
    axes_list[1].set_title("Model size distribution")

    for index, batch in enumerate(batch_sizes):
        axis = axes_list[2 + index]
        axis.hist(training_memory_map[batch], bins=100, color="black")
        axis.set_xlabel("Training memory (MB)")
        axis.set_ylabel("Frequency")
        axis.set_title(f"Training memory distribution (batch={batch})")

    plt.tight_layout()

    if fig_save_path:
        fig.savefig(fig_save_path, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if csv_path:
        df_data = {
            "param_count": param_counts,
            "model_size_mb": model_sizes_mb,
            "params": collected_params,
        }
        if len(batch_sizes) == 1:
            df_data["training_memory_mb"] = training_memory_map[batch_sizes[0]]
        else:
            for batch in batch_sizes:
                column_name = f"training_memory_mb_batch_{batch}"
                df_data[column_name] = training_memory_map[batch]
        df = pd.DataFrame(df_data)
        if len(batch_sizes) == 1:
            df = df.sort_values("training_memory_mb", ascending=False)
        else:
            memory_columns = [f"training_memory_mb_batch_{batch}" for batch in batch_sizes]
            df["max_training_memory_mb"] = df[memory_columns].max(axis=1)
            df = df.sort_values("max_training_memory_mb", ascending=False)
        df.to_csv(csv_path, index=False)

    if corr_csv_path:
        param_df = pd.DataFrame(collected_params)
        param_df["param_count"] = param_counts
        numeric_df = param_df.select_dtypes(include=[np.number])
        if "param_count" in numeric_df.columns:
            corr = (
                numeric_df.corr(method="spearman")["param_count"]
                .drop("param_count")
                .dropna()
                .sort_values(key=abs, ascending=False)
            )
            corr.to_csv(corr_csv_path)

    if oom_count:
        print(f"{RED}Skipped {oom_count} trial(s) due to ResourceExhaustedError.{RESET}")
        print(f"{RED}Skipped {internal_error_count} trial(s) due to InternalError.{RESET}")
        print(f"{RED}Skipped {unavailable_count} trial(s) due to UnavailableError.{RESET}")
        print(f"{RED}Skipped {scratch_error_count} trial(s) due to cuDNN scratch space error.{RESET}")


def set_user_attr_model_stats(
    trial: optuna.Trial,
    model: tf.keras.Model,
    bytes_per_param: int,
    batch_size: int,
    n_trials: int = 10000,
    device: Union[int, str] = "both",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Extract and return model statistics from the given Optuna trial.
    
    Statistics include:
        - Number of parameters
        - Model size in bits
        - FLOPs (floating-point operations)
        - MACs (multiply-accumulate operations)
        - Model summary
        - Average per-inference resource stats (before, current, delta for system RAM, GPU RAM, GPU usage %, CPU usage %)
        - Inference time
        - Average power consumption
        - Average energy consumption

    Args:
        trial (optuna.Trial): The Optuna trial object
        model (tf.keras.Model): The Keras model to analyze.
        policy (tf.keras.DTypePolicy): The precision policy used for the model.
        batch_size (int): The batch size to simulate for input.
        n_trials (int): Number of trials for power and energy measurement.
        device (int | str): GPU index to run the model on. Use ``-1``/``"cpu"`` for
            CPU measurements or ``"both"``/``"both:<index>"`` to profile CPU and GPU
            sequentially (GPU index defaults to 0). Defaults to ``"both"``.
        verbose (bool): If True, print detailed information.

    Returns:
        Dict[str, Any]: A dictionary containing model statistics and formatted strings
    """
    params = model.count_params()
    model_size_bytes = params * bytes_per_param
    flops = get_flops(model)
    macs = get_macs(model)
    model_summary = capture_model_summary(model)

    def _resolve_device_mode(device_spec: Union[int, str]) -> str:
        if isinstance(device_spec, str):
            normalized = device_spec.strip().lower()
            if normalized.startswith("both"):
                return "both"
            if normalized in {"cpu", "-1"}:
                return "cpu"
            return "gpu"
        return "cpu" if device_spec == -1 else "gpu"

    device_mode = _resolve_device_mode(device)

    resource_usage_raw, inference_time_raw = get_memory_and_time(
        model, batch_size=batch_size, device=device, verbose=verbose
    )
    usage_stats_raw = get_model_usage_stats(
        model, device=device, n_trials=n_trials, verbose=verbose
    )

    def _normalize_resource_usage(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict) and all(key in {"cpu", "gpu"} for key in raw.keys()):
            return {key: raw[key] for key in ("gpu", "cpu") if key in raw}
        target = "cpu" if device_mode == "cpu" else "gpu"
        return {target: raw}

    def _normalize_inference_time(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict) and all(key in {"cpu", "gpu"} for key in raw.keys()):
            return {key: raw[key] for key in ("gpu", "cpu") if key in raw}
        target = "cpu" if device_mode == "cpu" else "gpu"
        return {target: raw}

    def _normalize_usage_stats(raw: Any) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for key in ("gpu", "cpu"):
                if key not in raw:
                    continue
                value = raw[key]
                if isinstance(value, dict):
                    normalized[key] = {
                        "per_run_time": value.get("per_run_time", "Not measured"),
                        "avg_power": value.get("avg_power", "Not measured"),
                        "avg_energy": value.get("avg_energy", "Not measured"),
                    }
                else:
                    normalized[key] = {
                        "per_run_time": value,
                        "avg_power": "Not measured",
                        "avg_energy": "Not measured",
                    }
            return normalized
        per_run_time_value, avg_power_value, avg_energy_value = raw
        target = "cpu" if device_mode == "cpu" else "gpu"
        normalized[target] = {
            "per_run_time": per_run_time_value,
            "avg_power": avg_power_value,
            "avg_energy": avg_energy_value,
        }
        return normalized

    per_device_resource_usage = _normalize_resource_usage(resource_usage_raw)
    per_device_inference_time = _normalize_inference_time(inference_time_raw)
    per_device_usage_stats = _normalize_usage_stats(usage_stats_raw)

    ram_metrics = {"system_ram", "gpu_ram"}

    def _format_ram_display(value: float) -> str:
        raw_int = int(round(value))
        return f"{raw_int} B ({format_bytes(value)})"

    def _build_resource_views(metrics_value: Any) -> Tuple[Any, Any]:
        if isinstance(metrics_value, str):
            return metrics_value, metrics_value
        diff_payload: Dict[str, Any] = {}
        display_payload: Dict[str, Any] = {}
        for metric_name, metric_value in metrics_value.items():
            if isinstance(metric_value, str):
                diff_payload[metric_name] = metric_value
                if metric_name in ram_metrics:
                    display_payload[metric_name] = metric_value
                continue

            if isinstance(metric_value, dict) and metric_value.get("error"):
                error_text = str(metric_value.get("error", "Unknown error"))
                if not error_text.lower().startswith("error"):
                    error_text = f"Error: {error_text}"
                diff_payload[metric_name] = error_text
                if metric_name in ram_metrics:
                    display_payload[metric_name] = error_text
                continue

            component_diff = metric_value.get("difference")
            diff_payload[metric_name] = (
                component_diff if component_diff is not None else "Not measured"
            )

            if metric_name in ram_metrics:
                component_display: Dict[str, str] = {}
                for component_name in ("before", "current", "difference"):
                    component_value = metric_value.get(component_name)
                    if component_value is None:
                        component_display[component_name] = "Not measured"
                    elif isinstance(component_value, str):
                        component_display[component_name] = component_value
                    else:
                        component_display[component_name] = _format_ram_display(component_value)
                display_payload[metric_name] = component_display

        return diff_payload, display_payload

    resource_usage_diff_map: Dict[str, Any] = {}
    resource_usage_display_map: Dict[str, Any] = {}

    for device_label, metrics_value in per_device_resource_usage.items():
        diff_payload, display_payload = _build_resource_views(metrics_value)
        resource_usage_diff_map[device_label] = diff_payload
        if isinstance(display_payload, dict) and display_payload:
            resource_usage_display_map[device_label] = display_payload
        elif isinstance(display_payload, str):
            resource_usage_display_map[device_label] = display_payload

    def _metric_component_for_device(
        metrics_value: Any,
        metric_name: str,
        component_name: str,
    ) -> Union[str, int, float]:
        if isinstance(metrics_value, str):
            return metrics_value
        metric_block = metrics_value.get(metric_name, "Not measured")
        if isinstance(metric_block, str):
            return metric_block
        if isinstance(metric_block, dict) and metric_block.get("error"):
            error_text = str(metric_block.get("error", "Unknown error"))
            if not error_text.lower().startswith("error"):
                error_text = f"Error: {error_text}"
            return error_text
        component_value = metric_block.get(component_name)
        if component_value is None:
            return "Not measured"
        return component_value

    avg_power_map: Dict[str, Any] = {}
    avg_energy_map: Dict[str, Any] = {}
    per_run_time_map: Dict[str, Any] = {}
    for device_label, stats_payload in per_device_usage_stats.items():
        avg_power_map[device_label] = stats_payload.get("avg_power", "Not measured")
        avg_energy_map[device_label] = stats_payload.get("avg_energy", "Not measured")
        per_run_time_map[device_label] = stats_payload.get("per_run_time", "Not measured")

    if device_mode == "cpu":
        primary_order: Tuple[str, ...] = ("cpu", "gpu")
    elif device_mode == "gpu":
        primary_order = ("gpu", "cpu")
    else:
        primary_order = ("gpu", "cpu")

    def _pick_primary(container: Dict[str, Any]) -> Any:
        if not container:
            return "Not measured"
        for key in primary_order:
            if key in container:
                return container[key]
        return next(iter(container.values()))

    def _value_for(container: Dict[str, Any], key: str) -> Any:
        return container.get(key, "Not measured")

    multi_device = len(per_device_resource_usage) > 1

    resource_usage_primary = _pick_primary(per_device_resource_usage)
    resource_usage_diff_primary = _pick_primary(resource_usage_diff_map)
    resource_usage_display_primary = (
        _pick_primary(resource_usage_display_map) if resource_usage_display_map else {}
    )

    inference_time_primary = _pick_primary(per_device_inference_time)
    avg_power_primary = _pick_primary(avg_power_map)
    avg_energy_primary = _pick_primary(avg_energy_map)
    per_run_time_primary = _pick_primary(per_run_time_map)

    inference_time_gpu = _value_for(per_device_inference_time, "gpu")
    inference_time_cpu = _value_for(per_device_inference_time, "cpu")
    avg_power_gpu = _value_for(avg_power_map, "gpu")
    avg_power_cpu = _value_for(avg_power_map, "cpu")
    avg_energy_gpu = _value_for(avg_energy_map, "gpu")
    avg_energy_cpu = _value_for(avg_energy_map, "cpu")
    per_run_time_gpu = _value_for(per_run_time_map, "gpu")
    per_run_time_cpu = _value_for(per_run_time_map, "cpu")

    trial.set_user_attr("resource_usage", resource_usage_primary)
    trial.set_user_attr("resource_usage_diff", resource_usage_diff_primary)
    if resource_usage_display_primary not in (None, {}, "Not measured"):
        trial.set_user_attr("resource_usage_display", resource_usage_display_primary)
    trial.set_user_attr("resource_usage_details", per_device_resource_usage)
    trial.set_user_attr("resource_usage_diff_details", resource_usage_diff_map)
    trial.set_user_attr("resource_usage_display_details", resource_usage_display_map)
    for device_label in ("gpu", "cpu"):
        trial.set_user_attr(
            f"resource_usage_{device_label}",
            per_device_resource_usage.get(device_label, "Not measured"),
        )
        trial.set_user_attr(
            f"resource_usage_diff_{device_label}",
            resource_usage_diff_map.get(device_label, "Not measured"),
        )
        trial.set_user_attr(
            f"resource_usage_display_{device_label}",
            resource_usage_display_map.get(device_label, "Not measured"),
        )

    trial.set_user_attr("inference_time", inference_time_primary)
    trial.set_user_attr("inference_time_details", per_device_inference_time)

    for device_label, value in per_device_inference_time.items():
        trial.set_user_attr(f"inference_time_{device_label}", value)

    for device_label in ("gpu", "cpu"):
        if device_label not in per_device_inference_time:
            trial.set_user_attr(f"inference_time_{device_label}", "Not measured")

    trial.set_user_attr("avg_power", avg_power_primary)
    trial.set_user_attr("avg_energy", avg_energy_primary)
    trial.set_user_attr("per_run_time", per_run_time_primary)
    trial.set_user_attr("avg_power_details", avg_power_map)
    trial.set_user_attr("avg_energy_details", avg_energy_map)
    trial.set_user_attr("per_run_time_details", per_run_time_map)

    for device_label in per_device_usage_stats:
        trial.set_user_attr(f"avg_power_{device_label}", avg_power_map.get(device_label, "Not measured"))
        trial.set_user_attr(f"avg_energy_{device_label}", avg_energy_map.get(device_label, "Not measured"))
        trial.set_user_attr(
            f"per_run_time_{device_label}",
            per_run_time_map.get(device_label, "Not measured"),
        )

    for device_label in ("gpu", "cpu"):
        if device_label not in per_device_usage_stats:
            trial.set_user_attr(f"avg_power_{device_label}", "Not measured")
            trial.set_user_attr(f"avg_energy_{device_label}", "Not measured")
            trial.set_user_attr(f"per_run_time_{device_label}", "Not measured")

    for device_label, metrics_value in per_device_resource_usage.items():
        suffix = "" if not multi_device else ("" if device_label == "gpu" else f"_{device_label}")
        diff_entry = resource_usage_diff_map.get(device_label, "Not measured")
        display_entry = resource_usage_display_map.get(device_label)
        for metric_name in ("system_ram", "gpu_ram", "gpu_usage", "cpu_usage"):
            before_value = _metric_component_for_device(metrics_value, metric_name, "before")
            current_value = _metric_component_for_device(metrics_value, metric_name, "current")
            if isinstance(diff_entry, dict):
                diff_value = diff_entry.get(metric_name, "Not measured")
            else:
                diff_value = diff_entry

            attr_prefix = metric_name if not suffix else f"{metric_name}{suffix}"
            trial.set_user_attr(f"{attr_prefix}_before", before_value)
            trial.set_user_attr(f"{attr_prefix}_current", current_value)
            trial.set_user_attr(f"{attr_prefix}_diff", diff_value)

            if metric_name in ram_metrics:
                if isinstance(display_entry, dict):
                    metric_display_entry = display_entry.get(metric_name)
                    if isinstance(metric_display_entry, dict):
                        trial.set_user_attr(
                            f"{attr_prefix}_before_display",
                            metric_display_entry.get("before", "Not measured"),
                        )
                        trial.set_user_attr(
                            f"{attr_prefix}_current_display",
                            metric_display_entry.get("current", "Not measured"),
                        )
                        trial.set_user_attr(
                            f"{attr_prefix}_diff_display",
                            metric_display_entry.get("difference", "Not measured"),
                        )
                    elif isinstance(metric_display_entry, str):
                        trial.set_user_attr(f"{attr_prefix}_display", metric_display_entry)
                elif isinstance(display_entry, str):
                    trial.set_user_attr(f"{attr_prefix}_display", display_entry)

    resource_usage_result = resource_usage_primary
    resource_usage_diff_result = resource_usage_diff_primary
    resource_usage_display_result = resource_usage_display_primary
    inference_time_result = inference_time_primary
    avg_power_result = avg_power_primary
    avg_energy_result = avg_energy_primary
    per_run_time_result = per_run_time_primary

    def _format_metric_with_suffix(value: int, unit: str, joiner: str = "") -> str:
        formatted = format_number(value)
        if formatted.startswith("Invalid input"):
            return f"{value} {unit}"

        sign = ""
        body = formatted
        if formatted.startswith("-"):
            sign = "-"
            body = formatted[1:].lstrip()

        parts = body.split()
        if len(parts) == 2:
            magnitude, prefix = parts
            formatted_with_unit = f"{sign}{magnitude} {prefix}{joiner}{unit}".strip()
        else:
            formatted_with_unit = f"{formatted} {unit}".strip()

        return f"{value} {unit} ({formatted_with_unit})"

    def _format_bytes_with_suffix(value: int) -> str:
        human_readable = format_bytes(value)
        if human_readable.startswith("Invalid input"):
            return f"{value} B"
        return f"{value} B ({human_readable})"

    num_params_display = _format_metric_with_suffix(params, "parameters", joiner=" ")
    model_size_display = _format_bytes_with_suffix(model_size_bytes)
    flops_display = _format_metric_with_suffix(flops, "FLOPs")
    macs_display = _format_metric_with_suffix(macs, "MACs")

    trial.set_user_attr("num_params", params)
    trial.set_user_attr("num_params_display", num_params_display)
    trial.set_user_attr("model_size", model_size_bytes)
    trial.set_user_attr("model_size_display", model_size_display)
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("flops_display", flops_display)
    trial.set_user_attr("macs", macs)
    trial.set_user_attr("macs_display", macs_display)
    trial.set_user_attr("model_summary", model_summary)

    return {
        "num_params": params,
        "num_params_display": num_params_display,
        "model_size": model_size_bytes,
        "model_size_display": model_size_display,
        "flops": flops,
        "flops_display": flops_display,
        "macs": macs,
        "macs_display": macs_display,
        "model_summary": model_summary,
        "resource_usage": resource_usage_result,
        "resource_usage_diff": resource_usage_diff_result,
        "resource_usage_display": resource_usage_display_result,
        "resource_usage_details": per_device_resource_usage,
        "resource_usage_diff_details": resource_usage_diff_map,
        "resource_usage_display_details": resource_usage_display_map,
        "resource_usage_gpu": per_device_resource_usage.get("gpu", "Not measured"),
        "resource_usage_cpu": per_device_resource_usage.get("cpu", "Not measured"),
        "resource_usage_diff_gpu": resource_usage_diff_map.get("gpu", "Not measured"),
        "resource_usage_diff_cpu": resource_usage_diff_map.get("cpu", "Not measured"),
        "resource_usage_display_gpu": resource_usage_display_map.get("gpu", "Not measured"),
        "resource_usage_display_cpu": resource_usage_display_map.get("cpu", "Not measured"),
        "inference_time": inference_time_result,
        "inference_time_details": per_device_inference_time,
        "inference_time_gpu": inference_time_gpu,
        "inference_time_cpu": inference_time_cpu,
        "avg_power": avg_power_result,
        "avg_power_details": avg_power_map,
        "avg_power_gpu": avg_power_gpu,
        "avg_power_cpu": avg_power_cpu,
        "avg_energy": avg_energy_result,
        "avg_energy_details": avg_energy_map,
        "avg_energy_gpu": avg_energy_gpu,
        "avg_energy_cpu": avg_energy_cpu,
        "per_run_time": per_run_time_result,
        "per_run_time_details": per_run_time_map,
        "per_run_time_gpu": per_run_time_gpu,
        "per_run_time_cpu": per_run_time_cpu,
    }
