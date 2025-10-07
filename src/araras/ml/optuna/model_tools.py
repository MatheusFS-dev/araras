from araras.core import *

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
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

from araras.ml.model.stats import (
    get_flops,
    get_macs,
    get_memory_and_time,
    get_model_usage_stats,
)
from araras.ml.model.utils import capture_model_summary, parse_device_spec
from araras.utils.misc import format_number, format_bytes

from araras.visualization.configs import config_plt

from araras.ml.model.tools import save_model_plot

# ———————————————————————————————————————————————————————————————————————————— #
#                                   Utilities                                  #
# ———————————————————————————————————————————————————————————————————————————— #


def _dtype_size(dtype: Any, default: int = 4) -> int:
    """Return the storage size in bytes for a TensorFlow dtype.

    Args:
        dtype (Any): TensorFlow dtype or dtype-compatible spec.
        default (int): Fallback number of bytes to return when ``dtype`` cannot
            be resolved. Defaults to ``4`` (float32).

    Returns:
        int: Number of bytes required to store a scalar element with ``dtype``.

    Raises:
        ValueError: If ``default`` is negative.

    Notes:
        TensorFlow raises ``TypeError`` when attempting to convert an invalid
        dtype. Those errors are suppressed in favour of the provided ``default``
        value to keep the estimator resilient.

    Warnings:
        Passing custom dtypes that cannot be resolved by :func:`tf.as_dtype`
        forces the estimator to use ``default`` which might underestimate memory
        requirements.
    """

    if default < 0:
        raise ValueError("default must be non-negative")

    if dtype is None:
        return default

    try:
        return int(tf.as_dtype(dtype).size)
    except TypeError:
        try:
            return int(tf.as_dtype(str(dtype)).size)
        except (TypeError, ValueError):
            return default


def _resolve_layer_output_shape(layer: keras.layers.Layer) -> Optional[Any]:
    """Return the best-effort static output shape for a layer.

    Args:
        layer (keras.layers.Layer): Layer whose output shape should be resolved.

    Returns:
        Optional[Any]: A TensorFlow :class:`~tf.TensorShape`, a nested collection
        of shapes when the layer emits multiple tensors, or ``None`` when no
        static shape information is available.

    Notes:
        Starting with Keras 3, ``layer.output_shape`` often returns ``None`` even
        after the layer is connected to a computation graph. This helper
        inspects ``layer.output`` (which yields ``KerasTensor`` objects with
        populated ``shape`` metadata) and gracefully falls back to legacy
        attributes such as ``output_shape`` or ``get_output_shape_at`` when
        present.

    Warnings:
        Custom layers that override ``output`` with side effects or emit ragged
        tensors may still fail to provide a usable static shape. Callers should
        handle ``None`` results accordingly.
    """

    output_shape = getattr(layer, "output_shape", None)

    if output_shape is None and hasattr(layer, "get_output_shape_at"):
        try:
            output_shape = layer.get_output_shape_at(0)
        except Exception:
            output_shape = None

    if output_shape is None and hasattr(layer, "input_shape") and hasattr(layer, "compute_output_shape"):
        try:
            output_shape = layer.compute_output_shape(layer.input_shape)
        except Exception:
            output_shape = None

    if output_shape is None:
        output_tensor = getattr(layer, "output", None)
        if isinstance(output_tensor, (list, tuple)):
            shapes = [getattr(tensor, "shape", None) for tensor in output_tensor]
            if all(shape is not None for shape in shapes):
                output_shape = shapes
        elif output_tensor is not None:
            output_shape = getattr(output_tensor, "shape", None)

    return output_shape


def _count_params_and_bytes(weights: Iterable[tf.Variable]) -> Tuple[int, int]:
    """Count parameters and raw storage bytes for a list of variables.

    Args:
        weights (Iterable[tf.Variable]): Collection of TensorFlow variables whose
            parameter count and storage footprint should be measured.

    Returns:
        Tuple[int, int]: A tuple ``(param_count, byte_size)`` describing the
        total number of scalar parameters and the corresponding storage
        requirements.

    Notes:
        TensorFlow may lazily create optimizer slot variables. Ensure the model
        is compiled before calling this helper when slot sizes are required.

    Warnings:
        Mixed precision models that keep shadow copies in higher precision may
        produce additional variables outside ``model.trainable_weights``. Those
        are not accounted for here and should be handled separately if needed.
    """

    total_params = 0
    total_bytes = 0
    for weight in weights:
        params = int(tf.keras.backend.count_params(weight))
        total_params += params
        total_bytes += params * _dtype_size(getattr(weight, "dtype", None))
    return total_params, total_bytes


def _resolve_policy_dtypes(model: keras.Model) -> Tuple[int, int, int]:
    """Resolve dtype sizes (bytes) for variables, compute path and gradients.

    Args:
        model (keras.Model): Model whose dtype policy should be analysed.

    Returns:
        Tuple[int, int, int]: ``(variable_bytes, compute_bytes, gradient_bytes)``
        describing the dtype sizes used for weights, forward activations and
        gradients respectively.

    Notes:
        ``tf.keras.mixed_precision.Policy`` exposes both ``variable_dtype`` and
        ``compute_dtype`` which the estimator leverages to reason about mixed
        precision behaviour. Gradient tensors are assumed to use the widest
        precision among the variable dtype, compute dtype, and the backend
        default float type.

    Warnings:
        In custom training loops gradients may be cast to alternative dtypes.
        The estimator assumes gradients follow the compute dtype.
    """

    policy = getattr(model, "dtype_policy", None)

    variable_dtype_bytes = _dtype_size(
        getattr(policy, "variable_dtype", None) if policy is not None else getattr(model, "dtype", None)
    )
    compute_dtype_bytes = _dtype_size(
        getattr(policy, "compute_dtype", None) if policy is not None else getattr(model, "dtype", None),
        default=variable_dtype_bytes,
    )
    gradient_candidates = [compute_dtype_bytes, variable_dtype_bytes]
    gradient_candidates.append(
        _dtype_size(tf.keras.backend.floatx(), default=compute_dtype_bytes)
    )
    if policy is not None:
        gradient_candidates.append(
            _dtype_size(getattr(policy, "compute_dtype", None), default=compute_dtype_bytes)
        )
    gradient_dtype_bytes = max(gradient_candidates)
    return variable_dtype_bytes, compute_dtype_bytes, gradient_dtype_bytes


def _get_optimizer_slot_multiplier(model: keras.Model) -> int:
    """Infer the number of optimizer slot tensors per trainable weight.

    Args:
        model (keras.Model): Compiled model with an attached optimizer.

    Returns:
        int: Number of optimizer-maintained slot tensors per weight variable.

    Notes:
        Slot tensors store optimizer-specific statistics (for example, Adam's
        first and second moments). When the optimizer is unknown a conservative
        default of ``1`` slot per weight is used.

    Warnings:
        Custom optimizers may maintain additional tensors. For accurate results
        extend this helper accordingly.
    """

    if not hasattr(model, "optimizer") or model.optimizer is None:
        logger.warning(
            "Model has no compiled optimizer; assuming zero optimizer slot tensors."
        )
        return 0

    optimizer_name = model.optimizer.__class__.__name__.lower()

    if "adam" in optimizer_name or "adamax" in optimizer_name or "nadam" in optimizer_name:
        return 2  # m and v slots
    if "rmsprop" in optimizer_name:
        # Accumulator + optional momentum buffer
        has_momentum = getattr(model.optimizer, "momentum", 0) not in (0, None)
        return 2 if has_momentum else 1
    if "adagrad" in optimizer_name:
        return 1
    if "adadelta" in optimizer_name:
        return 2
    if "sgd" in optimizer_name:
        momentum = getattr(model.optimizer, "momentum", 0)
        return 1 if momentum and momentum > 0 else 0
    return 1  # sensible default if optimizer is unknown


def _shape_element_count(shape: Any) -> int:
    """Compute the element count for a tensor shape without the batch axis.

    Args:
        shape (Any): Shape descriptor coming from ``layer.output_shape`` or
            related Keras APIs.

    Returns:
        int: Number of scalar elements represented by ``shape`` ignoring the
        leading batch dimension.

    Raises:
        ValueError: If ``shape`` is not interpretable as a TensorFlow shape.

    Notes:
        When a dimension is ``None`` the estimator substitutes ``1`` to maintain
        a conservative footprint.

    Warnings:
        Shapes corresponding to ragged tensors are not supported and result in a
        ``ValueError``.
    """

    if shape is None:
        return 0

    if isinstance(shape, tf.TensorShape):
        if shape.rank is None:
            return 0
        shape_list = shape.as_list()
    else:
        if not isinstance(shape, (list, tuple)):
            raise ValueError("Unsupported shape specification encountered")
        shape_list = list(shape)

    if shape_list is None or not shape_list:
        return 0

    first_dim = shape_list[0]
    if isinstance(first_dim, (list, tuple, tf.TensorShape)):
        return sum(_shape_element_count(sub_shape) for sub_shape in shape_list)

    dims = [dim if (dim is not None and dim > 0) else 1 for dim in shape_list[1:]]
    if not dims:
        return 1
    return int(np.prod(dims))


def _calculate_activation_memory(
    model: keras.Model,
    batch_size: int,
    compute_dtype_bytes: int,
    gradient_dtype_bytes: int,
    trainable_params: int,
    *,
    verbose: bool = False,
) -> int:
    """Estimate activation memory using layer output shapes from the model summary.

    Args:
        model (keras.Model): Model whose activations should be analysed.
        batch_size (int): Training batch size to scale activation storage.
        compute_dtype_bytes (int): Number of bytes used for forward activations.
        gradient_dtype_bytes (int): Number of bytes used for gradient tensors.
        trainable_params (int): Count of trainable parameters, used for fallback
            heuristics.
        verbose (bool): Whether to emit detailed logging for intermediate
            results.

    Returns:
        int: Estimated activation memory in bytes for one training step.

    Notes:
        The estimator walks through ``model.layers`` and aggregates the output
        tensor sizes provided by Keras. The resulting value accounts for both the
        forward activations and their gradient counterparts.

    Warnings:
        Layers without a static ``output_shape`` trigger a parameter-based
        fallback which may under-estimate the true memory requirements.
    """

    per_sample_bytes = 0
    missing_layers: list[str] = []

    for layer in model.layers:
        output_shape = _resolve_layer_output_shape(layer)

        if output_shape is None:
            missing_layers.append(layer.name)
            continue

        try:
            elements = _shape_element_count(output_shape)
        except ValueError:
            missing_layers.append(layer.name)
            continue
        if elements == 0:
            missing_layers.append(layer.name)
            continue

        layer_dtype_bytes = _dtype_size(
            getattr(layer, "compute_dtype", None),
            default=_dtype_size(getattr(layer, "dtype", None), default=compute_dtype_bytes),
        )
        per_sample_bytes += elements * layer_dtype_bytes

    if per_sample_bytes == 0:
        if verbose:
            if missing_layers:
                logger.warning(
                    "Falling back to parameter-based activation estimate; missing output shapes for layers: %s",
                    ", ".join(sorted(set(missing_layers))),
                )
            else:
                logger.warning(
                    "Falling back to parameter-based activation estimate; no activation sizes available."
                )
        return int(trainable_params * gradient_dtype_bytes * 1.5 * batch_size)

    if verbose:
        logger.info("Activation memory per sample: %s", format_bytes(per_sample_bytes))

    return int(per_sample_bytes * batch_size * 2)


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


def estimate_training_memory(model: keras.Model, batch_size: int = 32, verbose: int = 0) -> int:
    """Estimate the training memory footprint for a compiled Keras model.

    The estimator inspects the model summary to derive parameter counts, per-layer
    activation sizes, and optimizer state requirements. Optimizer slot variables
    (for example, Adam's first and second moment estimates) are inferred from the
    optimizer type. When detailed layer information is not available the function
    falls back to parameter-based heuristics and emits a warning.

    Args:
        model (keras.Model): The compiled Keras model to analyse.
        batch_size (int): Mini-batch size used for the training step simulation.
        verbose (int): Controls logging verbosity. ``1`` enables detailed logging
            through :data:`araras.core.logger`, while ``0`` silences it. Only
            ``0`` and ``1`` are supported.

    Returns:
        int: Estimated peak memory usage in bytes required to train ``model``.

    Raises:
        ValueError: If ``verbose`` is not ``0`` or ``1``.

    Notes:
        The returned value includes model weights, gradient tensors, optimizer
        slots, activation buffers for the forward/backward passes, and a
        framework-dependent overhead term. Actual device usage can still differ
        based on TensorFlow runtime optimisations and operator-level
        implementations.

    Warnings:
        Estimations rely on static layer metadata. Models featuring dynamic
        control flow, ragged tensors, or custom layers without ``output_shape``
        information can trigger fallback heuristics that may under-estimate the
        real peak memory usage.
    """

    if verbose not in (0, 1):
        raise ValueError("verbose must be 0 or 1")

    log_enabled = bool(verbose)

    def _log_info(message: str, *args: Any) -> None:
        if log_enabled:
            logger.info(message, *args)

    _log_info(
        "Estimating training memory for model '%s' with batch size %d.",
        getattr(model, "name", "<unnamed>"),
        batch_size,
    )

    variable_dtype_bytes, compute_dtype_bytes, gradient_dtype_bytes = _resolve_policy_dtypes(model)
    _log_info(
        "Dtype sizes (variable/compute/gradient): %d / %d / %d bytes",
        variable_dtype_bytes,
        compute_dtype_bytes,
        gradient_dtype_bytes,
    )

    trainable_params, trainable_bytes = _count_params_and_bytes(model.trainable_weights)
    non_trainable_params, non_trainable_bytes = _count_params_and_bytes(model.non_trainable_weights)

    model_size_bytes = trainable_bytes + non_trainable_bytes
    gradient_bytes = trainable_params * gradient_dtype_bytes

    slot_multiplier = _get_optimizer_slot_multiplier(model)
    optimizer_slot_bytes = trainable_params * gradient_dtype_bytes * slot_multiplier

    activation_memory = _calculate_activation_memory(
        model,
        batch_size,
        compute_dtype_bytes,
        gradient_dtype_bytes,
        trainable_params,
        verbose=log_enabled,
    )

    framework_overhead = _get_framework_overhead()

    total_memory = model_size_bytes + gradient_bytes + optimizer_slot_bytes + activation_memory + framework_overhead

    _log_info(
        "Trainable params: %s (%s)",
        format_number(trainable_params),
        format_bytes(trainable_bytes),
    )
    _log_info(
        "Non-trainable params: %s (%s)",
        format_number(non_trainable_params),
        format_bytes(non_trainable_bytes),
    )
    _log_info("Model size (weights): %s", format_bytes(model_size_bytes))
    _log_info("Gradient memory: %s", format_bytes(gradient_bytes))
    _log_info("Optimizer slots (%d per weight): %s", slot_multiplier, format_bytes(optimizer_slot_bytes))
    _log_info("Activation memory (batch): %s", format_bytes(activation_memory))
    _log_info("Framework overhead: %s", format_bytes(framework_overhead))
    _log_info("Total estimated memory: %s", format_bytes(total_memory))

    return int(total_memory)


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
    verbose: bool = False,
) -> None:
    """Sample random models, plot statistics and optionally save the results.

    This helper draws ``n_trials`` random models using ``build_model_fn`` and records
    their parameter counts, approximate model sizes, and estimated training memory
    consumption. Histograms for each metric can be saved to disk and, optionally,
    displayed after sampling completes. TensorFlow sessions are cleared between
    trials to release GPU memory. Trials that raise
    :class:`tf.errors.ResourceExhaustedError`, :class:`tf.errors.InternalError`,
    :class:`tf.errors.UnavailableError`, or cuDNN scratch-space allocation errors are
    skipped because they typically indicate out-of-memory conditions. The number of
    skipped trials is counted and printed at the end. When ``csv_path`` is provided,
    the collected statistics are saved to a CSV file, including each trial's
    parameters and sorted in decreasing order of the estimated training memory. If
    ``logs_dir`` is set, parameters of failed trials along with the error traceback
    are saved to individual log files.

    Args:
        build_model_fn (Callable[[optuna.Trial], tf.keras.Model]): Callable that receives an
            Optuna :class:`optuna.Trial` and returns a compiled
            :class:`tf.keras.Model`.
        bytes_per_param (int): Number of bytes used to store each parameter.
        batch_size (Union[int, Iterable[int]]): Batch size used when estimating the
            training memory. Can be a single int or an iterable of ints; when
            multiple values are provided they are all evaluated.
        n_trials (int): Total number of random trials to sample.
        fig_save_path (Optional[str]): Optional path to save the figure. If ``None``
            and ``show_plot`` is ``True``, the figure is only displayed.
        figsize (Tuple[int, int]): Figure size for the histograms.
        csv_path (Optional[str]): Optional path to store trial results as CSV. The
            CSV includes the sampled parameters and is sorted by
            ``training_memory_mb`` in descending order.
        logs_dir (Optional[str]): Directory where error logs are written. If
            ``None``, no logs are saved.
        corr_csv_path (Optional[str]): Optional path to store correlations between
            numeric hyperparameters and the model parameter count. If ``None`` the
            correlation analysis is skipped.
        plot_model_dir (Optional[str]): Directory where model plots are saved. If
            ``None``, no plots are saved. Each model is saved as a PNG file named
            ``model_{trial_number}.png``.
        show_plot (bool): Whether to display the histogram figure after sampling.
            Defaults to ``False``. When ``False`` the figure backend is switched to
            a non-interactive renderer to avoid X11 allocation errors and the plot
            is only saved.
        verbose (bool): If ``True``, print detailed information during sampling.

    Returns:
        None: This function performs its work for side effects and returns ``None``.

    Raises:
        ValueError: If ``batch_size`` is provided as an empty iterable.
        TypeError: If ``batch_size`` contains non-integer values.

    Notes:
        Clearing the Keras backend session between trials mitigates
        :class:`tf.errors.ResourceExhaustedError` on GPUs with limited VRAM. When
        ``logs_dir`` is provided, one log file per failed trial is created
        containing the trial parameters and traceback. Graphviz must be installed
        to plot models.

    Warnings:
        Models that trigger :class:`tf.errors.ResourceExhaustedError` are ignored in
        the final statistics.
    """
    if not show_plot:
        try:
            plt.switch_backend("Agg")
        except Exception:
            # Fallback silently if the backend cannot be switched (e.g., already in use).
            pass

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
            description="Sampling trials",
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
            
            size_mb = (n_params * bytes_per_param) / (1024 * 1024)
            model_sizes_mb.append(size_mb)

            for batch in batch_sizes:
                training_memory_mb = (
                    estimate_training_memory(model, batch_size=batch, verbose=verbose) / (1024 * 1024)
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
        try:
            plt.show()
        except Exception as exc:
            print(
                (
                    f"{YELLOW}Warning: Unable to display the Optuna search-space plot "
                    f"due to: {exc}. Common causes include running without an available "
                    "X11 display, insufficient pixmap memory (e.g., BadAlloc), or using "
                    "monitor on a headless server. Consider re-running with show_plot="
                    "False, launching a virtual display (such as Xvfb), or reducing the "
                    "figure DPI before displaying plots.{RESET}"
                )
            )
            raise
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
    device: str = "both/0",
    stats_to_measure: Iterable[str] = (
        "parameters",
        "flops",
        "macs",
        "summary",
        "resource_usage",
        "usage_stats",
    ),
    verbose: bool = False,
) -> Dict[str, Any]:
    """Collect model statistics and store them as Optuna user attributes.

    The helper estimates parameter counts, model size, FLOPs, MACs, model
    summaries, resource usage, inference time, and usage statistics (power,
    energy, and per-run time). Callers can limit the amount of profiling work by
    explicitly selecting which statistic groups to measure through
    ``stats_to_measure``. Any disabled group is reported back as skipped to make
    omissions explicit in user attributes and the returned payload.

    Args:
        trial (optuna.Trial): Optuna trial that will receive user attributes.
        model (tf.keras.Model): Keras model whose statistics are extracted.
        bytes_per_param (int): Storage size for each parameter, in bytes.
        batch_size (int): Mini-batch size used during resource profiling.
        n_trials (int): Number of inference runs used to estimate usage stats.
        device (str): Canonical device specifier. Use ``"cpu"`` for CPU-only
            profiling, ``"gpu/<index>"`` for a single GPU, or ``"both/<index>"`` to
            profile CPU and GPU ``<index>`` sequentially. Defaults to ``"both/0"``.
        stats_to_measure (Iterable[str]): Iterable describing which statistic
            groups to compute. Use ``"parameters"`` to gather parameter counts
            and serialized size estimates, ``"flops"`` for floating-point
            operation counts, ``"macs"`` for multiply-accumulate estimates,
            ``"summary"`` to capture the ``model.summary`` output,
            ``"resource_usage"`` for exclusive RAM/VRAM consumption and
            inference latency, and ``"usage_stats"`` for power, energy, and
            per-run timing metrics gathered from resource monitoring. Any
            iterable omitting one or more of these values skips the respective
            measurement.
        verbose (bool): When ``True``, print detailed progress information during
            profiling.

    Returns:
        Dict[str, Any]: Dictionary containing the collected statistics and the
        formatted strings stored in trial attributes.

    Raises:
        TypeError: If ``stats_to_measure`` is ``None`` or not iterable.
        ValueError: If ``stats_to_measure`` includes values outside the supported
            set or ``device`` is not expressed as ``"cpu"``, ``"gpu/<index>"``, or
            ``"both/<index>"``.

    Notes:
        Skipped statistics return ``None`` for raw numeric values and
        ``"Not measured (skipped)"`` for their formatted display strings to make
        omissions explicit.

    Warnings:
        Measuring resource usage and usage statistics can significantly increase
        runtime for large models. Restrict ``stats_to_measure`` to the required
        groups when profiling time is a concern.
    """

    supported_stats: Tuple[str, ...] = (
        "parameters",
        "flops",
        "macs",
        "summary",
        "resource_usage",
        "usage_stats",
    )
    allowed_stats = set(supported_stats)
    selected_stats: List[str] = []
    invalid_entries: List[str] = []

    if stats_to_measure is None:
        raise TypeError("stats_to_measure must be an iterable of statistic names")

    stats_iterable = stats_to_measure

    for raw_entry in stats_iterable:
        normalized_entry = str(raw_entry).strip().lower()
        if not normalized_entry:
            continue
        if normalized_entry not in allowed_stats:
            invalid_entries.append(str(raw_entry))
            continue
        if normalized_entry not in selected_stats:
            selected_stats.append(normalized_entry)

    if invalid_entries:
        valid_options = ", ".join(supported_stats)
        raise ValueError(
            "Unsupported statistics requested: "
            f"{invalid_entries}. Valid options are: {valid_options}."
        )

    measure_parameters = "parameters" in selected_stats
    measure_flops = "flops" in selected_stats
    measure_macs = "macs" in selected_stats
    measure_summary = "summary" in selected_stats
    measure_resource_usage = "resource_usage" in selected_stats
    measure_usage_stats = "usage_stats" in selected_stats

    not_measured = "Not measured"
    skipped_text = "Not measured (skipped)"

    params: Optional[int] = None
    model_size_bytes: Optional[int] = None
    if measure_parameters:
        params = model.count_params()
        model_size_bytes = params * bytes_per_param

    flops: Optional[int] = get_flops(model) if measure_flops else None
    macs: Optional[int] = get_macs(model) if measure_macs else None
    model_summary = (
        capture_model_summary(model) if measure_summary else skipped_text
    )

    device_kind, _ = parse_device_spec(device)
    device_mode = device_kind

    per_device_resource_usage: Dict[str, Any] = {}
    per_device_inference_time: Dict[str, Any] = {}
    resource_usage_diff_map: Dict[str, Any] = {}
    resource_usage_display_map: Dict[str, Any] = {}
    resource_usage_primary: Any = not_measured
    resource_usage_diff_primary: Any = not_measured
    resource_usage_display_primary: Any = {}
    inference_time_primary: Any = not_measured
    inference_time_gpu: Any = not_measured
    inference_time_cpu: Any = not_measured
    multi_device = False
    default_resource_text = not_measured
    default_inference_text = not_measured

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
                        "per_run_time": value.get("per_run_time", not_measured),
                        "avg_power": value.get("avg_power", not_measured),
                        "avg_energy": value.get("avg_energy", not_measured),
                    }
                else:
                    normalized[key] = {
                        "per_run_time": value,
                        "avg_power": not_measured,
                        "avg_energy": not_measured,
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

            delta_stats = metric_value.get("delta") if isinstance(metric_value, dict) else None
            delta_max = None
            if isinstance(delta_stats, dict):
                delta_max = delta_stats.get("max")
            diff_payload[metric_name] = (
                delta_max if delta_max is not None else not_measured
            )

            if metric_name in ram_metrics:
                component_display: Dict[str, str] = {}
                for component_name in ("before", "during", "delta"):
                    component_stats = (
                        metric_value.get(component_name)
                        if isinstance(metric_value, dict)
                        else None
                    )
                    if not isinstance(component_stats, dict):
                        component_display[component_name] = not_measured
                        continue

                    component_max = component_stats.get("max")
                    if component_max is None:
                        component_display[component_name] = not_measured
                    else:
                        component_display[component_name] = _format_ram_display(component_max)
                display_payload[metric_name] = component_display

        return diff_payload, display_payload

    if measure_resource_usage:
        resource_usage_raw, inference_time_raw = get_memory_and_time(
            model, batch_size=batch_size, device=device, verbose=verbose
        )
        per_device_resource_usage = _normalize_resource_usage(resource_usage_raw)
        per_device_inference_time = _normalize_inference_time(inference_time_raw)
        for device_label, metrics_value in per_device_resource_usage.items():
            diff_payload, display_payload = _build_resource_views(metrics_value)
            resource_usage_diff_map[device_label] = diff_payload
            if isinstance(display_payload, dict) and display_payload:
                resource_usage_display_map[device_label] = display_payload
            elif isinstance(display_payload, str):
                resource_usage_display_map[device_label] = display_payload
        multi_device = len(per_device_resource_usage) > 1
    else:
        resource_usage_primary = skipped_text
        resource_usage_diff_primary = skipped_text
        resource_usage_display_primary = {}
        inference_time_primary = skipped_text
        inference_time_gpu = skipped_text
        inference_time_cpu = skipped_text
        default_resource_text = skipped_text
        default_inference_text = skipped_text

    def _metric_component_for_device(
        metrics_value: Any,
        metric_name: str,
        component_name: str,
    ) -> Union[str, int, float, None]:
        if isinstance(metrics_value, str):
            return metrics_value
        metric_block = metrics_value.get(metric_name, not_measured)
        if isinstance(metric_block, str):
            return metric_block
        if isinstance(metric_block, dict) and metric_block.get("error"):
            error_text = str(metric_block.get("error", "Unknown error"))
            if not error_text.lower().startswith("error"):
                error_text = f"Error: {error_text}"
            return error_text
        target_stats = metric_block.get(component_name)
        if not isinstance(target_stats, dict):
            return not_measured
        value = target_stats.get("max")
        return not_measured if value is None else value

    if measure_usage_stats:
        usage_stats_raw = get_model_usage_stats(
            model, device=device, n_trials=n_trials, verbose=verbose
        )
        per_device_usage_stats = _normalize_usage_stats(usage_stats_raw)
    else:
        per_device_usage_stats = {}

    avg_power_map: Dict[str, Any] = {}
    avg_energy_map: Dict[str, Any] = {}
    per_run_time_map: Dict[str, Any] = {}
    for device_label, stats_payload in per_device_usage_stats.items():
        avg_power_map[device_label] = stats_payload.get("avg_power", not_measured)
        avg_energy_map[device_label] = stats_payload.get("avg_energy", not_measured)
        per_run_time_map[device_label] = stats_payload.get("per_run_time", not_measured)

    if device_mode == "cpu":
        primary_order: Tuple[str, ...] = ("cpu", "gpu")
    elif device_mode == "gpu":
        primary_order = ("gpu", "cpu")
    else:
        primary_order = ("gpu", "cpu")

    def _pick_primary(container: Dict[str, Any]) -> Any:
        if not container:
            return not_measured
        for key in primary_order:
            if key in container:
                return container[key]
        return next(iter(container.values()))

    def _value_for(container: Dict[str, Any], key: str) -> Any:
        return container.get(key, not_measured)

    if measure_resource_usage:
        resource_usage_primary = _pick_primary(per_device_resource_usage)
        resource_usage_diff_primary = _pick_primary(resource_usage_diff_map)
        resource_usage_display_primary = (
            _pick_primary(resource_usage_display_map) if resource_usage_display_map else {}
        )
        inference_time_primary = _pick_primary(per_device_inference_time)
        inference_time_gpu = _value_for(per_device_inference_time, "gpu")
        inference_time_cpu = _value_for(per_device_inference_time, "cpu")

    avg_power_primary = _pick_primary(avg_power_map)
    avg_energy_primary = _pick_primary(avg_energy_map)
    per_run_time_primary = _pick_primary(per_run_time_map)

    avg_power_gpu = _value_for(avg_power_map, "gpu")
    avg_power_cpu = _value_for(avg_power_map, "cpu")
    avg_energy_gpu = _value_for(avg_energy_map, "gpu")
    avg_energy_cpu = _value_for(avg_energy_map, "cpu")
    per_run_time_gpu = _value_for(per_run_time_map, "gpu")
    per_run_time_cpu = _value_for(per_run_time_map, "cpu")

    default_usage_text = not_measured
    if not measure_usage_stats:
        avg_power_primary = skipped_text
        avg_energy_primary = skipped_text
        per_run_time_primary = skipped_text
        avg_power_gpu = skipped_text
        avg_power_cpu = skipped_text
        avg_energy_gpu = skipped_text
        avg_energy_cpu = skipped_text
        per_run_time_gpu = skipped_text
        per_run_time_cpu = skipped_text
        default_usage_text = skipped_text

    trial.set_user_attr("resource_usage", resource_usage_primary)
    trial.set_user_attr("resource_usage_diff", resource_usage_diff_primary)
    if resource_usage_display_primary not in (None, {}, not_measured, skipped_text):
        trial.set_user_attr("resource_usage_display", resource_usage_display_primary)
    trial.set_user_attr("resource_usage_details", per_device_resource_usage)
    trial.set_user_attr("resource_usage_diff_details", resource_usage_diff_map)
    trial.set_user_attr("resource_usage_display_details", resource_usage_display_map)
    for device_label in ("gpu", "cpu"):
        trial.set_user_attr(
            f"resource_usage_{device_label}",
            per_device_resource_usage.get(device_label, default_resource_text),
        )
        trial.set_user_attr(
            f"resource_usage_diff_{device_label}",
            resource_usage_diff_map.get(device_label, default_resource_text),
        )
        trial.set_user_attr(
            f"resource_usage_display_{device_label}",
            resource_usage_display_map.get(device_label, default_resource_text),
        )

    trial.set_user_attr("inference_time", inference_time_primary)
    trial.set_user_attr("inference_time_details", per_device_inference_time)

    for device_label, value in per_device_inference_time.items():
        trial.set_user_attr(f"inference_time_{device_label}", value)

    for device_label in ("gpu", "cpu"):
        if device_label not in per_device_inference_time:
            trial.set_user_attr(f"inference_time_{device_label}", default_inference_text)

    trial.set_user_attr("avg_power", avg_power_primary)
    trial.set_user_attr("avg_energy", avg_energy_primary)
    trial.set_user_attr("per_run_time", per_run_time_primary)
    trial.set_user_attr("avg_power_details", avg_power_map)
    trial.set_user_attr("avg_energy_details", avg_energy_map)
    trial.set_user_attr("per_run_time_details", per_run_time_map)

    for device_label in per_device_usage_stats:
        trial.set_user_attr(
            f"avg_power_{device_label}",
            avg_power_map.get(device_label, not_measured),
        )
        trial.set_user_attr(
            f"avg_energy_{device_label}",
            avg_energy_map.get(device_label, not_measured),
        )
        trial.set_user_attr(
            f"per_run_time_{device_label}",
            per_run_time_map.get(device_label, not_measured),
        )

    for device_label in ("gpu", "cpu"):
        if device_label not in per_device_usage_stats:
            trial.set_user_attr(f"avg_power_{device_label}", default_usage_text)
            trial.set_user_attr(f"avg_energy_{device_label}", default_usage_text)
            trial.set_user_attr(f"per_run_time_{device_label}", default_usage_text)

    for device_label, metrics_value in per_device_resource_usage.items():
        suffix = "" if not multi_device else ("" if device_label == "gpu" else f"_{device_label}")
        diff_entry = resource_usage_diff_map.get(device_label, default_resource_text)
        display_entry = resource_usage_display_map.get(device_label)
        for metric_name in ("system_ram", "gpu_ram", "gpu_usage", "cpu_usage"):
            before_value = _metric_component_for_device(metrics_value, metric_name, "before")
            current_value = _metric_component_for_device(metrics_value, metric_name, "during")
            if isinstance(diff_entry, dict):
                diff_value = diff_entry.get(metric_name, default_resource_text)
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
                            metric_display_entry.get("before", not_measured),
                        )
                        trial.set_user_attr(
                            f"{attr_prefix}_current_display",
                            metric_display_entry.get("during", not_measured),
                        )
                        trial.set_user_attr(
                            f"{attr_prefix}_diff_display",
                            metric_display_entry.get("delta", not_measured),
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

    if params is not None:
        num_params_display = _format_metric_with_suffix(params, "parameters", joiner=" ")
    else:
        num_params_display = skipped_text

    if model_size_bytes is not None:
        model_size_display = _format_bytes_with_suffix(model_size_bytes)
    else:
        model_size_display = skipped_text

    if flops is not None:
        flops_display = _format_metric_with_suffix(flops, "FLOPs")
    else:
        flops_display = skipped_text

    if macs is not None:
        macs_display = _format_metric_with_suffix(macs, "MACs")
    else:
        macs_display = skipped_text

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
        "resource_usage_gpu": per_device_resource_usage.get("gpu", default_resource_text),
        "resource_usage_cpu": per_device_resource_usage.get("cpu", default_resource_text),
        "resource_usage_diff_gpu": resource_usage_diff_map.get("gpu", default_resource_text),
        "resource_usage_diff_cpu": resource_usage_diff_map.get("cpu", default_resource_text),
        "resource_usage_display_gpu": resource_usage_display_map.get("gpu", default_resource_text),
        "resource_usage_display_cpu": resource_usage_display_map.get("cpu", default_resource_text),
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
