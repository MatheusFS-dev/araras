from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import optuna
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gc
import pandas as pd
import numpy as np
import traceback
import os

from araras.ml.model.stats import (
    get_flops,
    get_macs,
    get_model_stats,
    render_model_stats_report,
)
from araras.ml.model.utils import capture_model_summary, parse_device_spec
from araras.utils.misc import format_number, format_bytes
from araras.visualization.configs import config_plt
from araras.ml.model.tools import save_model_plot
from araras.utils.verbose_printer import VerbosePrinter
from araras.utils.loading_bar import gen_loading_bar

vp = VerbosePrinter()


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
    gradient_candidates.append(_dtype_size(tf.keras.backend.floatx(), default=compute_dtype_bytes))
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
        vp.printf(
            f"Model has no compiled optimizer; assuming zero optimizer slot tensors.",
            tag="[ARARAS WARNING] ",
            color="yellow",
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
) -> int:
    """Estimate activation memory using layer output shapes from the model summary.

    Args:
        model (keras.Model): Model whose activations should be analysed.
        batch_size (int): Training batch size to scale activation storage.
        compute_dtype_bytes (int): Number of bytes used for forward activations.
        gradient_dtype_bytes (int): Number of bytes used for gradient tensors.
        trainable_params (int): Count of trainable parameters, used for fallback
            heuristics.

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
        return int(trainable_params * gradient_dtype_bytes * 1.5 * batch_size)

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


def estimate_training_memory(model: keras.Model, batch_size: int = 32) -> int:
    """Estimate the training memory footprint for a compiled Keras model.

    The estimator inspects the model summary to derive parameter counts, per-layer
    activation sizes, and optimizer state requirements. Optimizer slot variables
    (for example, Adam's first and second moment estimates) are inferred from the
    optimizer type. When detailed layer information is not available the function
    falls back to parameter-based heuristics and emits a warning.

    Args:
        model (keras.Model): The compiled Keras model to analyse.
        batch_size (int): Mini-batch size used for the training step simulation.

    Returns:
        int: Estimated peak memory usage in bytes required to train ``model``.


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
    variable_dtype_bytes, compute_dtype_bytes, gradient_dtype_bytes = _resolve_policy_dtypes(model)

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
    )

    framework_overhead = _get_framework_overhead()

    total_memory = (
        model_size_bytes + gradient_bytes + optimizer_slot_bytes + activation_memory + framework_overhead
    )

    return int(total_memory)


def prune_model_by_config(
    trial: optuna.Trial,
    model: keras.Model,
    thresholds: Dict[str, float],
    *,
    bytes_per_param: int = 8,
    batch_size: int = 1,
    verbose: int = 0,
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
        verbose (int): If greater than ``0``, print detailed logging through the process.

    Raises:
        optuna.TrialPruned: If any threshold is exceeded.

    Notes:
        When multiple thresholds are provided the model will be pruned as soon
        as the first limit is violated.

    Warnings:
        The memory estimation relies on :func:`estimate_training_memory` and is
        therefore only an approximation.
    """
    vp.verbose = verbose

    # Do nothing if user pass thresholds={}
    if not thresholds:
        return

    metrics = {
        "param": model.count_params(),
        "model_size": model.count_params() * bytes_per_param / (1024 * 1024),
        "memory_mb": estimate_training_memory(model, batch_size=batch_size) / (1024 * 1024),
        "flops": get_flops(model, batch_size=1),
    }

    if verbose > 0:
        vp.printf(f"Model statistics for trial {trial.number}:", tag="[ARARAS INFO] ", color="blue")
        for key, value in metrics.items():
            vp.printf(vp.color(f"  {key}: {value:.2f}", "blue"))
        print()

    for key, threshold in thresholds.items():
        value = metrics.get(key)
        if value is None:
            continue
        if value > threshold:
            vp.printf(
                f"Pruning trial {trial.number}: {key} {value:.2f} exceeds {threshold}",
                tag="[ARARAS WARNING] ",
                color="yellow",
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
    verbose: int = 1,
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
        verbose (int): If greater than ``0``, print detailed information during sampling.

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
        progress_iter = gen_loading_bar(
            progress_iter,
            description=vp.color("Sampling models", "blue"),
            total=n_trials,
            bar_color="blue",
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
                training_memory_mb = estimate_training_memory(model, batch_size=batch, verbose=verbose) / (
                    1024 * 1024
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
                    vp.printf(
                        f"Failed to plot model {trial.number}: {e}", tag="[ARARAS ERROR] ", color="red"
                    )
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
            vp.printf(
                (
                    f"Unable to display the Optuna search-space plot due to: {exc}."
                    " Common causes include running without an available X11 display,"
                    " insufficient pixmap memory (e.g., BadAlloc), or using monitor on a"
                    " headless server. Consider re-running with show_plot=False, launching"
                    " a virtual display (such as Xvfb), or reducing the figure DPI before"
                    " displaying plots."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
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
        vp.printf(f"Skipped {oom_count} trial(s) due to ResourceExhaustedError.", tag="[ARARAS WARNING] ", color="orange")
        vp.printf(f"Skipped {internal_error_count} trial(s) due to InternalError.", tag="[ARARAS WARNING] ", color="orange")
        vp.printf(f"Skipped {unavailable_count} trial(s) due to UnavailableError.", tag="[ARARAS WARNING] ", color="orange")
        vp.printf(f"Skipped {scratch_error_count} trial(s) due to cuDNN scratch space error.", tag="[ARARAS WARNING] ", color="orange")



def set_user_attr_model_stats(
    trial: optuna.Trial,
    model: tf.keras.Model,
    *,
    batch_size: int = 1,
    device: str = "both/0",
    stats_to_measure: Iterable[str] = (
        "parameters",
        "model_size",
        "flops",
        "macs",
        "summary",
        "inference_latency",
        "cpu_util_percent",
        "cpu_power_rapl_w",
        "ram_used_bytes",
        "ram_util_percent",
        "gpu_util_percent",
        "gpu_mem_used_bytes",
        "gpu_power_w",
    ),
    test_runs: int = 10,
    verbose: int = 1,
    bytes_per_param: int = 4,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Profile a model and persist statistics on an Optuna trial.

    This helper mirrors :func:`araras.ml.model.stats.get_model_stats` by
    computing structural metrics (parameter count, model size, FLOPs, MACs,
    and summary text) alongside runtime measurements (latency and resource
    utilisation). The collected information is stored as user attributes on
    ``trial`` and returned for immediate consumption.

    Args:
        trial (optuna.Trial): Optuna trial that will receive the statistics as
            user attributes.
        model (tf.keras.Model): Compiled Keras model to profile.
        batch_size (int): Batch size forwarded to :func:`get_model_stats` when
            generating dummy inputs. Defaults to ``1``.
        device (str): Canonical device selector. Use ``"cpu"`` for CPU-only
            profiling, ``"gpu/<index>"`` for a specific GPU, or
            ``"both/<index>"`` to execute CPU and GPU profiling sequentially.
            Defaults to ``"both/0"``.
        stats_to_measure (Iterable[str]): Iterable of metric identifiers
            forwarded to :func:`get_model_stats`. Defaults to the standard
            metric list.
        test_runs (int): Number of repetitions used when sampling resource
            metrics. Defaults to ``10``.
        verbose (int): Verbosity level forwarded to :func:`get_model_stats`.
            Defaults to ``1``.
        bytes_per_param (int): Number of bytes associated with each
            parameter when estimating the ``model_size`` metric. Defaults to
            ``4``.
        extra_attrs (Optional[Dict[str, Any]]): Additional attributes appended
            to the generated textual report and stored as user attributes.
            Defaults to ``None``.

    Returns:
        Dict[str, Optional[Dict[str, Any]]]: Mapping with possible ``"cpu"``
        and ``"gpu"`` entries containing the statistics collected for each
        device.

    Raises:
        ValueError: If ``bytes_per_param`` is less than ``1`` or the device
            specification is invalid.
        RuntimeError: Propagated when the requested GPU index is unavailable
            on the current system.

    Notes:
        The helper stores both the raw metric dictionaries and human-readable
        renderings (such as ``num_params_display``) to preserve backwards
        compatibility with existing tooling. Missing metrics are recorded as
        ``"N/A"`` strings in the formatted attributes.

    Warnings:
        Profiling allocates temporary tensors on the selected devices. Large
        models may require substantial memory during the measurement phase,
        potentially leading to ``ResourceExhaustedError`` if resources are
        constrained.
    """

    if bytes_per_param < 1:
        raise ValueError("bytes_per_param must be at least 1")

    extra_attrs = extra_attrs or {}

    stats_kwargs = {
        "batch_size": batch_size,
        "stats_to_measure": stats_to_measure,
        "test_runs": test_runs,
        "verbose": verbose,
        "bytes_per_param": bytes_per_param,
    }

    device_kind, gpu_index = parse_device_spec(device)

    cpu_stats: Optional[Dict[str, Any]] = None
    gpu_stats: Optional[Dict[str, Any]] = None

    if device_kind == "both":
        if gpu_index is None:
            raise ValueError("GPU device index could not be resolved for combined profiling")
        gpu_device = f"gpu/{gpu_index}"
        gpu_stats = get_model_stats(model, device=gpu_device, **stats_kwargs)
        cpu_stats = get_model_stats(model, device="cpu", **stats_kwargs)
    elif device_kind == "gpu":
        if gpu_index is None:
            raise ValueError("GPU device index could not be resolved")
        gpu_device = f"gpu/{gpu_index}"
        gpu_stats = get_model_stats(model, device=gpu_device, **stats_kwargs)
    else:
        cpu_stats = get_model_stats(model, device="cpu", **stats_kwargs)

    stats_map: Dict[str, Dict[str, Any]] = {}
    if gpu_stats:
        stats_map["gpu"] = gpu_stats
        trial.set_user_attr("model_stats_gpu", gpu_stats)
    if cpu_stats:
        stats_map["cpu"] = cpu_stats
        trial.set_user_attr("model_stats_cpu", cpu_stats)

    trial.set_user_attr("model_stats", stats_map)

    structural_stats = next((stats for stats in (gpu_stats, cpu_stats) if stats), {})

    def _format_with_unit(value: Optional[int], unit: str, formatter: Callable[[int], str]) -> Optional[str]:
        if value is None:
            return None
        human = formatter(value)
        return f"{value} {unit} ({human})"

    def _format_bytes_value(value: Optional[int]) -> Optional[str]:
        if value is None:
            return None
        return f"{value} B ({format_bytes(value)})"

    num_params = structural_stats.get("parameters") if structural_stats else None
    model_size = structural_stats.get("model_size") if structural_stats else None
    flops = structural_stats.get("flops") if structural_stats else None
    macs = structural_stats.get("macs") if structural_stats else None
    summary = structural_stats.get("summary") if structural_stats else None

    num_params_display = _format_with_unit(num_params, "parameters", format_number)
    model_size_display = _format_bytes_value(model_size)
    flops_display = _format_with_unit(flops, "FLOPs", format_number)
    macs_display = _format_with_unit(macs, "MACs", format_number)

    trial.set_user_attr("num_params", num_params)
    trial.set_user_attr("num_params_display", num_params_display or "N/A")
    trial.set_user_attr("model_size", model_size)
    trial.set_user_attr("model_size_display", model_size_display or "N/A")
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("flops_display", flops_display or "N/A")
    trial.set_user_attr("macs", macs)
    trial.set_user_attr("macs_display", macs_display or "N/A")
    trial.set_user_attr("model_summary", summary)

    report_text = render_model_stats_report(
        structural_stats,
        cpu_stats=cpu_stats,
        gpu_stats=gpu_stats,
        extra_attrs=extra_attrs,
    )
    trial.set_user_attr("model_stats_report", report_text)
    if extra_attrs:
        trial.set_user_attr("model_stats_extra_attrs", extra_attrs)

    return {"gpu": gpu_stats, "cpu": cpu_stats}
