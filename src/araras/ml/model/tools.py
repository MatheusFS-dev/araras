import math
from typing import Any, Dict, Literal, Sequence, Union


import tempfile
import zipfile
import traceback

from pathlib import Path
from araras.ml.model.stats import get_flops

import tensorflow as tf

from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

_VALIDATE_STEPS_EXECUTED_ONCE = False
_VALIDATE_STEPS_LAST_RESULT: Union[Dict[str, Any], None] = None


def convert_to_saved_model(input_keras_path: str, output_zip_path: str) -> None:
    """Convert a `.keras` archive into a zipped TensorFlow SavedModel.

    Args:
        input_keras_path (str): Path to the source ``.keras`` model file.
        output_zip_path (str): Destination path for the resulting ``.zip`` bundle.

    Returns:
        None: The SavedModel artefacts are written into ``output_zip_path``.

    Raises:
        Exception: Propagates loader or saver failures raised by TensorFlow.
    """
    # Create a temp workspace for the SavedModel directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = Path(tmp_dir) / "saved_model"

        # Load the single-file Keras archive (.keras)
        model = tf.keras.models.load_model(input_keras_path)

        # Export the model in SavedModel format (creates a folder with saved_model.pb, variables/, assets/)
        tf.saved_model.save(model, str(saved_model_dir))

        # Compress the entire SavedModel directory tree into a zip file
        with zipfile.ZipFile(output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in saved_model_dir.rglob("*"):
                # Compute the archive name by stripping off the temp-dir prefix
                arcname = file_path.relative_to(saved_model_dir.parent)
                zf.write(file_path, arcname)


def validate_steps_per_execution(
    data_size: Union[int, Sequence[int]],
    batch_size: Union[int, Sequence[int]],
    steps_per_execution: Union[int, Sequence[int]],
    name: Union[str, Sequence[str]] = "Dataset",
    top_k_recommendations: int = 5,
    execute_once: bool = True,
) -> Dict[str, Any]:
    """
    Validate that steps_per_execution does not exceed the number of batches.
    Compute per-dataset execution metrics and recommend candidate values.
    When multiple datasets are provided, recommendations are unified into a
    single shared ``steps_per_execution`` ranking.

    Args:
        data_size (int | Sequence[int]): Number of samples in the dataset. Accepts a single value or a list/tuple.
        batch_size (int | Sequence[int]): Batch size for training. Accepts a single value or a list/tuple aligned to ``data_size``.
        steps_per_execution (int | Sequence[int]): Number of steps per execution. Accepts a single value or a list/tuple aligned to ``data_size``.
        name (str | Sequence[str], optional): Dataset name(s) for logging. Defaults to "Dataset".
        top_k_recommendations (int, optional): Maximum number of recommendations
            returned in the final ranking. Defaults to ``5``.
        execute_once (bool, optional): If ``True``, run the computation only on
            the first call and skip subsequent calls. Defaults to ``True``.

    Returns:
        dict[str, Any]: Validation payload with per-dataset metrics and a single
            recommendation ranking.

    Raises:
        TypeError: If numeric fields are not integer-like.
        ValueError: If ``steps_per_execution`` exceeds total batches, values are invalid,
            or input shapes mismatch.
    """
    global _VALIDATE_STEPS_EXECUTED_ONCE, _VALIDATE_STEPS_LAST_RESULT

    if not isinstance(execute_once, bool):
        raise TypeError("execute_once must be a boolean")
    if execute_once and _VALIDATE_STEPS_EXECUTED_ONCE:
        # vp.printf(
        #     "validate_steps_per_execution skipped: execute_once=True and it already ran.",
        #     tag="[ARARAS] ",
        #     color="yellow",
        # )
        return _VALIDATE_STEPS_LAST_RESULT or {
            "datasets": [],
            "recommendation_scope": "skipped",
            "recommendations": [],
            "skipped": True,
        }

    def _normalize_to_list(value, target_len, label):
        if isinstance(value, (list, tuple)):
            values = list(value)
            if len(values) != target_len:
                raise ValueError(f"{label} must have length {target_len} when provided as a list or tuple")
            return values
        return [value] * target_len

    def _require_int(value, label):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        return int(value)

    def _near_divisors(n, pivot, limit=8):
        if n <= 0:
            return []
        divisors = set()
        root = int(math.sqrt(n))
        for i in range(1, root + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        ordered = sorted(divisors, key=lambda d: (abs(d - pivot), -d))
        return ordered[:limit]

    data_is_list = isinstance(data_size, (list, tuple))
    name_is_list = isinstance(name, (list, tuple))

    if data_is_list != name_is_list:
        raise ValueError("data_size and name must both be scalars or both be lists/tuples of the same length")
    if data_is_list:
        data_sizes = list(data_size)
        names = list(name)
        if len(data_sizes) != len(names):
            raise ValueError("data_size and name must have the same length when provided as lists or tuples")
    else:
        data_sizes = [data_size]
        names = [name]

    entries = len(data_sizes)
    batch_sizes = _normalize_to_list(batch_size, entries, "batch_size")
    steps_list = _normalize_to_list(steps_per_execution, entries, "steps_per_execution")

    top_k_recommendations = _require_int(top_k_recommendations, "top_k_recommendations")
    if top_k_recommendations <= 0:
        raise ValueError("top_k_recommendations must be positive")

    dataset_metrics = []
    dataset_reco_inputs = []
    for ds, bs, spe, label in zip(data_sizes, batch_sizes, steps_list, names):
        ds = _require_int(ds, "data_size")
        bs = _require_int(bs, "batch_size")
        spe = _require_int(spe, "steps_per_execution")

        if ds < 0:
            raise ValueError("data_size cannot be negative")
        if ds == 0:
            raise ValueError("data_size cannot be zero because no batches can be formed")
        if bs <= 0:
            raise ValueError("batch_size must be positive")
        if spe <= 0:
            raise ValueError("steps_per_execution must be positive")

        total_batches = (ds + bs - 1) // bs
        if spe > total_batches:
            raise ValueError(
                f"{label} steps per exec ({spe}) exceeds total batches ({total_batches}). "
                "Reduce steps per exec or increase batch size."
            )

        executions_per_epoch = (total_batches + spe - 1) // spe
        tail_steps = total_batches % spe
        effective_slots = executions_per_epoch * spe
        utilization = total_batches / effective_slots
        padding_steps = effective_slots - total_batches

        # Heuristic target: balance callback frequency and host/device overhead.
        target_executions = max(1, min(32, int(round(math.sqrt(total_batches)))))
        ideal_spe = max(1, min(total_batches, (total_batches + target_executions - 1) // target_executions))

        dataset_metrics.append(
            {
                "name": label,
                "data_size": ds,
                "batch_size": bs,
                "total_batches": total_batches,
                "steps_per_execution": spe,
                "executions_per_epoch": executions_per_epoch,
                "tail_steps": tail_steps,
                "padding_steps": padding_steps,
                "utilization": utilization,
                "target_executions": target_executions,
                "ideal_steps_per_execution": ideal_spe,
            }
        )
        dataset_reco_inputs.append(
            {
                "total_batches": total_batches,
                "target_executions": target_executions,
                "ideal_steps_per_execution": ideal_spe,
            }
        )

    min_total_batches = min(item["total_batches"] for item in dataset_metrics)
    ideal_center = max(
        1,
        min(
            min_total_batches,
            int(round(sum(item["ideal_steps_per_execution"] for item in dataset_reco_inputs) / len(dataset_reco_inputs))),
        ),
    )

    candidate_values = {1, min_total_batches, ideal_center}
    for spe in steps_list:
        if spe <= min_total_batches:
            candidate_values.add(int(spe))

    for power in range(1, 30):
        val = 2**power
        if val > min_total_batches:
            break
        candidate_values.add(val)

    for exec_target in (1, 2, 4, 8, 16, 24, 32):
        candidate_values.add(max(1, min(min_total_batches, (min_total_batches + exec_target - 1) // exec_target)))

    for div in _near_divisors(min_total_batches, ideal_center, limit=12):
        candidate_values.add(div)

    candidate_values = sorted(c for c in candidate_values if 1 <= c <= min_total_batches)

    global_recommendations = []
    for candidate in candidate_values:
        per_dataset = []
        total_padding = 0
        total_utilization = 0.0
        total_execs = 0.0
        max_tail = 0
        score_acc = 0.0

        for info in dataset_reco_inputs:
            tb = info["total_batches"]
            execs = (tb + candidate - 1) // candidate
            tail = tb % candidate
            slots = execs * candidate
            padding = slots - tb
            util = tb / slots
            waste = 1.0 - util
            overhead_distance = abs(execs - info["target_executions"]) / max(info["target_executions"], 1)
            score_acc += waste + (0.45 * overhead_distance)

            total_padding += padding
            total_utilization += util
            total_execs += execs
            max_tail = max(max_tail, tail)
            per_dataset.append(
                {
                    "total_batches": tb,
                    "executions_per_epoch": execs,
                    "tail_steps": tail,
                    "padding_steps": padding,
                    "utilization": util,
                }
            )

        avg_utilization = total_utilization / len(dataset_reco_inputs)
        avg_execs = total_execs / len(dataset_reco_inputs)
        score = score_acc / len(dataset_reco_inputs)

        global_recommendations.append(
            {
                "steps_per_execution": candidate,
                "avg_executions_per_epoch": avg_execs,
                "avg_utilization": avg_utilization,
                "total_padding_steps": total_padding,
                "max_tail_steps": max_tail,
                "score": score,
                "per_dataset": per_dataset,
            }
        )

    global_recommendations.sort(
        key=lambda item: (
            item["score"],
            item["total_padding_steps"],
            abs(item["steps_per_execution"] - ideal_center),
        )
    )
    global_recommendations = global_recommendations[:top_k_recommendations]

    vp.printf("steps_per_execution validation summary", tag="[ARARAS] ", color="blue")
    summary_header = (
        f"{'Dataset':<16} {'Samples':>10} {'Batch':>8} {'Batches':>8} "
        f"{'SPE':>6} {'Exec/Epoch':>11} {'Tail':>6} {'Util':>8}"
    )
    vp.printf(summary_header, tag="[ARARAS] ", color="blue")
    vp.printf("-" * len(summary_header), tag="[ARARAS] ", color="blue")
    for item in dataset_metrics:
        vp.printf(
            (
                f"{str(item['name'])[:16]:<16} {item['data_size']:>10} {item['batch_size']:>8} "
                f"{item['total_batches']:>8} {item['steps_per_execution']:>6} "
                f"{item['executions_per_epoch']:>11} {item['tail_steps']:>6} "
                f"{item['utilization']:>7.2%}"
            ),
            tag="[ARARAS] ",
            color="blue",
        )

    scope_label = "global" if entries > 1 else "single-dataset"
    vp.printf(
        f"recommended {scope_label} steps_per_execution (top {top_k_recommendations})",
        tag="[ARARAS] ",
        color="blue",
    )
    reco_header = (
        f"{'Rank':>4} {'SPE':>6} {'AvgExec':>8} {'AvgUtil':>8} "
        f"{'MaxTail':>8} {'PadSum':>8}"
    )
    vp.printf(reco_header, tag="[ARARAS] ", color="blue")
    vp.printf("-" * len(reco_header), tag="[ARARAS] ", color="blue")
    for i, item in enumerate(global_recommendations, start=1):
        vp.printf(
            (
                f"{i:>4} {item['steps_per_execution']:>6} {item['avg_executions_per_epoch']:>8.2f} "
                f"{item['avg_utilization']:>7.2%} {item['max_tail_steps']:>8} "
                f"{item['total_padding_steps']:>8}"
            ),
            tag="[ARARAS] ",
            color="blue",
        )

    result = {
        "datasets": dataset_metrics,
        "recommendation_scope": scope_label,
        "recommendations": global_recommendations,
    }
    if execute_once:
        _VALIDATE_STEPS_EXECUTED_ONCE = True
        _VALIDATE_STEPS_LAST_RESULT = result

    return result


def save_model_plot(
    model_or_path: Union[tf.keras.Model, str, Path],
    output_path: Union[str, Path],
    safe_mode: bool = True,
) -> None:
    """
    Save a visual representation of a Keras model architecture to an image file.

    This function can take either a Keras model instance or a path to a `.keras`
    model archive. It generates a plot of the model architecture and saves it
    to the specified output path.

    Notes:
        The resulting plot format is inferred from the file extension provided
        in ``output_path``. A common choice is ``.png``. Existing
        files at ``output_path`` will be overwritten.

    Args:
        model_or_path (Union[tf.keras.Model, str, Path]): Either a :class:`tf.keras.Model` instance to be plotted or
            a filesystem path pointing to a ``.keras`` model archive.
        output_path (Union[str, Path]): Destination path where the plot image will be saved.
        safe_mode (bool): a parameter from :func:`tf.keras.utils.plot_model` that
            controls whether to use safe mode for plotting. If set to `True`, the
            function will not plot layers that are not supported by the plotting
            utility, which can help avoid errors with custom layers.

    Returns:
        None: The plot is written to ``output_path``.

    Raises:
        TypeError: If ``model_or_path`` is neither a model instance nor a path.
        ValueError: If a path is provided that does not end with ``.keras``.
        OSError: If saving the generated plot fails.
    """

    if isinstance(model_or_path, tf.keras.Model):
        model = model_or_path
    elif isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path)
        if model_path.suffix != ".keras":
            raise ValueError("Model path must point to a '.keras' file.")
        try:
            model = tf.keras.models.load_model(model_path, safe_mode=safe_mode)
        except Exception as exc:  # pragma: no cover - external load
            raise OSError("Failed to load Keras model from file") from exc
    else:
        raise TypeError("model_or_path must be a tf.keras.Model or path to a '.keras' file")

    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )
    except Exception as exc:
        vp.printf(f"\nFailed to save model plot: {exc}", tag="[ARARAS ERROR] ", color="red")
        vp.printf(f"Ensure 'graphviz' is installed and updated.", tag="[ARARAS WARNING] ", color="yellow")
        vp.printf(f"If using conda, try: conda install graphviz python-graphviz", tag="[ARARAS WARNING] ", color="yellow")

        traceback.print_exc()


def punish_model_flops(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-10,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's FLOPs.

    Args:
        target (Union[float, Sequence[float]]): Base objective value (scalar or list of scalars).
        model (tf.keras.Model): Model whose FLOPs will be used for the penalty.
        penalty_factor (float): Multiplicative factor applied to the FLOPs count.
        direction (Literal['minimize', 'maximize']): Whether the objective should be minimised or maximised.

    Returns:
        Union[float, Sequence[float]]: The penalised objective value or list of values.
    """

    if direction not in ("minimize", "maximize"):
        raise ValueError("`direction` must be either 'minimize' or 'maximize'.")

    total_flops = get_flops(model)

    # Compute penalty
    penalty = penalty_factor * total_flops

    # Apply penalty to single value or list
    if isinstance(target, (list, tuple)):
        return [t + penalty if direction == "minimize" else t - penalty for t in target]
    return target + penalty if direction == "minimize" else target - penalty


def punish_model_params(
    target: Union[float, Sequence[float]],
    model: tf.keras.Model,
    penalty_factor: float = 1e-9,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> Union[float, Sequence[float]]:
    """Penalize an objective according to the model's parameter count.

    Args:
        target (Union[float, Sequence[float]]): Base objective value (scalar or list of scalars).
        model (tf.keras.Model): Model whose parameters will be used for the penalty.
        penalty_factor (float): Multiplicative factor applied to the parameter count.
        direction (Literal['minimize', 'maximize']): Whether the objective should be minimised or maximised.

    Returns:
        Union[float, Sequence[float]]: The penalised objective value or list of values.
    """

    if direction not in ("minimize", "maximize"):
        raise ValueError("`direction` must be either 'minimize' or 'maximize'.")

    # Count total trainable + non-trainable parameters
    total_params = model.count_params()

    # Compute penalty
    penalty = penalty_factor * total_params

    # Apply penalty to single value or list
    if isinstance(target, (list, tuple)):
        return [t + penalty if direction == "minimize" else t - penalty for t in target]
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
        target (Union[float, Sequence[float]]): Base objective value (scalar or list of scalars).
        model (tf.keras.Model): Model whose complexity will be penalised.
        type (Literal['flops', 'params', None]): Type of penalty to apply, either "flops" or "params".
        flops_penalty_factor (float): Factor for FLOPs penalty.
        params_penalty_factor (float): Factor for parameters penalty.
        direction (Literal['minimize', 'maximize']): Whether the objective should be minimised or maximised.

    Returns:
        Union[float, Sequence[float]]: The penalised objective value or list of values.
    """
    if type is None:
        # If no type is specified, return the target unchanged
        return target

    if type == "flops":
        target = punish_model_flops(target, model, flops_penalty_factor, direction)
    elif type == "params":
        target = punish_model_params(target, model, params_penalty_factor, direction)
    else:
        raise ValueError("`type` must be either 'flops', 'params' or None.")

    return target


def print_tensor_mem(x, batch_size=None, name=None):
    """Print estimated memory usage for a tensor or layer output.

    Accepts a tensor, list/tuple of tensors, or a Keras layer (uses ``layer.output``).
    If the leading dimension is ``None`` and ``batch_size`` is provided, that value
    replaces the ``None`` to estimate batch memory. Useful for quick logging while
    building or debugging models. This is only a rough estimate of the tensor buffer
    itself; it does not account for model weights, intermediate activations, gradients,
    optimizer state, format conversions, or framework/runtime overhead.

    Args:
        x: Tensor, list/tuple of tensors, or ``tf.keras.layers.Layer``.
        batch_size (int | None): Optional batch size to substitute for an unknown
            leading dimension.
        name (str | None): Label to include in the printed output. Defaults to the
            tensor name when available.

    Usage:
        print_tensor_mem(model.output, batch_size=32, name="logits")
        print_tensor_mem(some_layer, batch_size=64)
    """

    def _prod(shape):
        p = 1
        for d in shape:
            if d is None:
                return None
            p *= int(d)
        return p

    def _mib(nbytes):
        if nbytes is None:
            return "unknown"
        return f"{nbytes / (1024**2):.3f} MiB"

    # Accept Layer or Tensor (KerasTensor)
    if isinstance(x, tf.keras.layers.Layer):
        if not hasattr(x, "output") or x.output is None:
            raise ValueError("Layer has no .output yet. Call it on an input first.")
        x = x.output  # may be tensor or list of tensors
        if name is None:
            name = x.name if hasattr(x, "name") else None

    def _one(t):
        shape = tuple(t.shape)
        if batch_size is not None and shape and shape[0] is None:
            shape = (batch_size,) + shape[1:]
        dtype = tf.as_dtype(t.dtype)
        n = _prod(shape)
        nbytes = None if n is None else n * dtype.size
        label = name or getattr(t, "name", "tensor")
        print(f"{label}: shape={shape}, dtype={dtype.name}, mem={_mib(nbytes)}")

    if isinstance(x, (list, tuple)):
        for i, t in enumerate(x):
            old = name
            name = f"{old}[{i}]" if old else None
            _one(t)
    else:
        _one(x)
