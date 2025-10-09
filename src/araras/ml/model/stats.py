from typing import Any, Dict, Iterable, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from araras.ml.model.utils import (
    capture_model_summary,
    parse_device_spec,
    run_dummy_inference,
)
from araras.utils.loading_bar import gen_loading_bar
from araras.utils.resource_monitor import ResourceMonitor
from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

RESOURCE_METRIC_AGGREGATIONS: Dict[str, str] = {
    "cpu_util_percent": "delta",
    "cpu_power_rapl_w": "peak",
    "ram_used_bytes": "delta",
    "ram_util_percent": "delta",
    "gpu_util_percent": "delta",
    "gpu_mem_used_bytes": "delta",
    "gpu_power_w": "peak",
}

GPU_ONLY_METRICS = {
    "gpu_util_percent",
    "gpu_mem_used_bytes",
    "gpu_power_w",
}


def get_flops(model: tf.keras.Model, batch_size: int = 1) -> int:
    """
    Calculate the floating-point operations (FLOPs) for a forward pass.
    This uses legacy TensorFlow profiler APIs, that is, tensorflow v1.

    Args:
        model (tf.keras.Model): Keras model to profile.
        batch_size (int): Batch size used to construct the dummy input tensors.

    Returns:
        int: Total number of FLOPs executed during a single forward pass.
    """

    # 1) Build one TensorSpec per input tensor
    specs = []
    for inp in model.inputs:
        # K.int_shape(inp) → (None, d1, d2, …)
        dims = K.int_shape(inp)[1:]  # drop the None batch dim
        specs.append(tf.TensorSpec([batch_size, *dims], dtype=inp.dtype))

    # 2) Define a wrapper whose args exactly match model.inputs
    @tf.function(input_signature=specs)
    def _forward_fn(*args):
        # args is a tuple of Tensors, one per input.
        # Pass them to the model as a list:
        return model(list(args), training=False)

    # 3) Grab the concrete graph and profile it
    concrete = _forward_fn.get_concrete_function()
    opts = ProfileOptionBuilder.float_operation()
    opts["output"] = "none"  # Supress report
    info = profile(concrete.graph, options=opts)
    return info.total_float_ops


def get_macs(model: tf.keras.Model, batch_size: int = 1) -> int:
    """Estimate multiply-accumulate operations (MACs) for a forward pass.

    Args:
        model (tf.keras.Model): Keras model to profile.
        batch_size (int): Batch size used to construct the dummy input tensors.

    Returns:
        int: Estimated number of MACs executed during a single forward pass.
    """

    return get_flops(model, batch_size) // 2


def get_inference_latency(
    model: tf.keras.Model,
    batch_size: int = 1,
    device: str = "cpu",
    *,
    warmup_runs: Optional[int] = None,
    runs: int = 1,
    verbose: int = 1,
) -> Tuple[float, float]:
    """
    Execute dummy inference passes on ``model`` and time them.

    The helper creates zero-filled tensors matching ``model.inputs`` for the
    requested ``batch_size`` and runs the model repeatedly on the selected
    device. Optional warm-up executions may be performed before timing begins
    to exclude one-off initialisation overheads. Each measured run converts the
    outputs to NumPy arrays to synchronise the execution graph and capture the
    true latency.

    Args:
        model (tf.keras.Model): Model whose inference latency should be
            measured.
        batch_size (int): Batch size for the dummy inputs. Defaults to ``1``.
        device (str): Device specification. Accepts ``"cpu"`` or
            ``"gpu/<index>"``. ``"both"`` is not supported. Defaults to
            ``"cpu"``.
        warmup_runs (Optional[int]): Number of warm-up executions performed
            before timing. ``None`` disables warm-ups. Defaults to ``None``.
        runs (int): Number of timed executions. Must be positive. Defaults to
            ``1``.
        verbose (int): Verbosity level. Values greater than zero render a
            progress bar. Defaults to ``1``.

    Returns:
        Tuple[float, float]: Average and peak inference latency in seconds.

    Raises:
        ValueError: If ``runs`` is less than ``1``, if ``batch_size`` is less
            than ``1``, or if ``device`` resolves to ``"both"``.
        RuntimeError: If the requested GPU device is unavailable.

    Notes:
        The model is executed inside a TensorFlow device context matching the
        requested ``device``. When a GPU is selected TensorFlow must have a
        visible physical GPU with the given index.
    """

    # This is just a proxy to run_dummy_inference with the same signature
    return run_dummy_inference(
        model,
        batch_size=batch_size,
        device=device,
        warmup_runs=warmup_runs,
        runs=runs,
        verbose=verbose,
    )


def get_model_stats(
    model: tf.keras.Model,
    batch_size: int = 1,
    device: str = "gpu/0",
    *,
    stats_to_measure: Iterable[str] = (
        "parameters",
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
) -> Dict[str, Any]:
    """Collect structural and runtime statistics for a Keras model.

    The helper aggregates structural information (parameter count, FLOPs, MACs,
    and summary) alongside runtime metrics derived from dummy inference runs.
    Latency calculations rely on :func:`run_dummy_inference`, while resource
    utilisation readings use :class:`ResourceMonitor` configured with short
    sampling windows. Resource metrics are recomputed ``test_runs`` times to
    smooth noisy measurements.

    Args:
        model (tf.keras.Model): Model to profile.
        batch_size (int): Batch size for dummy inputs. Defaults to ``1``.
        device (str): Target device string. Accepts ``"cpu"`` or
            ``"gpu/<index>"``. Defaults to ``"cpu"``.
        stats_to_measure (Iterable[str]): Iterable of metric identifiers to
            compute. Defaults to :data:`DEFAULT_STATS_TO_MEASURE`.
        test_runs (int): Number of repetitions for each resource metric.
            Defaults to ``10``.
        verbose (int): Verbosity level. Values above ``0`` render progress
            bars. Defaults to ``1``.

    Returns:
        Dict[str, Any]: Mapping from metric names to their computed statistics.
        Latency metrics return a mapping with ``average_s`` and ``peak_s``.
        Resource metrics provide aggregation metadata, the collected
        measurements, and simple summary statistics. Metrics that cannot be
        computed return ``None`` or an error string.

    Raises:
        ValueError: If ``device`` resolves to ``"both"`` or if ``test_runs`` is
            less than ``1``.
        RuntimeError: If the requested GPU index is unavailable on the current
            system.

    Notes:
        GPU-specific metrics are skipped when ``device`` targets the CPU. The
        function emits separate progress bars per metric when ``verbose`` is
        positive, including latency calculations inside
        :func:`run_dummy_inference`.

    Warnings:
        Resource measurements depend on system interfaces such as Intel RAPL
        and NVIDIA NVML. Missing interfaces yield ``None`` readings without
        raising exceptions.
    """

    if test_runs < 1:
        raise ValueError("test_runs must be at least 1")

    # —————————————————————————————— Device Handling ————————————————————————————— #
    device_kind, gpu_index = parse_device_spec(device)
    if device_kind == "both":
        raise ValueError("device must be either 'cpu' or 'gpu/<index>'")

    if device_kind == "gpu":
        if gpu_index is None:
            raise ValueError("GPU device index could not be resolved")
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus or gpu_index >= len(gpus):
            raise RuntimeError(f"No GPU found for index {gpu_index}")
    else:
        gpu_index = None

    resolved_gpu_index = gpu_index if gpu_index is not None else 0
    device_label = "CPU" if device_kind == "cpu" else f"GPU:{gpu_index}"

    # —————————————————————————— Setup for Measurements —————————————————————————— #
    stats_sequence = list(dict.fromkeys(stats_to_measure))
    results: Dict[str, Any] = {}

    def _measure_resource_metric(metric_name: str) -> Any:
        if metric_name in GPU_ONLY_METRICS and device_kind != "gpu":
            return None

        aggregation = RESOURCE_METRIC_AGGREGATIONS.get(metric_name)
        if aggregation is None:
            return None

        measurements: List[float] = []
        error_message: Optional[str] = None

        iterator: Iterable[int]
        iterator = range(test_runs)
        if verbose > 0:
            iterator = gen_loading_bar(
                iterator,
                description=vp.color(f"Measuring {metric_name} on {device_label}", "blue"),
                total=test_runs,
                bar_color="blue",
            )

        def _monitored_run() -> None:
            run_dummy_inference(
                model,
                batch_size=batch_size,
                device=device,
                warmup_runs=None,
                runs=1,
                verbose=0,
            )

        for _ in iterator:
            monitor = ResourceMonitor(
                {metric_name: aggregation},
                before_repetitions=1,
                during_repetitions=1,
                sample_interval_s=0.25,
                gpu_index=resolved_gpu_index,
                verbose=verbose > 1,
            )
            try:
                result = monitor.run_and_measure(_monitored_run)
            except Exception as exc:  # pragma: no cover - defensive guard
                error_message = f"Error ({exc.__class__.__name__}): {exc}"
                break
            value = result.get(metric_name)
            if value is not None:
                measurements.append(float(value))

        if error_message is not None:
            return error_message
        if not measurements:
            return None

        average = sum(measurements) / len(measurements)
        summary = {
            "aggregation": aggregation,
            "measurements": measurements,
            "average": average,
            "min": min(measurements),
            "max": max(measurements),
            "peak": max(measurements),
        }
        return summary

    # ———————————————————— Calculate each requested statistic ———————————————————— #
    for stat in stats_sequence:
        if stat == "parameters":
            results[stat] = int(model.count_params())
        elif stat == "flops":
            results[stat] = int(get_flops(model, batch_size))
        elif stat == "macs":
            results[stat] = int(get_macs(model, batch_size))
        elif stat == "summary":
            results[stat] = capture_model_summary(model)
        elif stat == "inference_latency":
            average, peak = run_dummy_inference(
                model,
                batch_size=batch_size,
                device=device,
                warmup_runs=10,
                runs=20,
                verbose=verbose,
            )
            results[stat] = {"average_s": average, "peak_s": peak}
        elif stat in RESOURCE_METRIC_AGGREGATIONS:
            results[stat] = _measure_resource_metric(stat)
        else:
            raise ValueError(
                f"Unknown statistic '{stat}' requested\nAccepted values: {list(RESOURCE_METRIC_AGGREGATIONS.keys()) + ['parameters', 'flops', 'macs', 'summary', 'inference_latency']}"
            )

    return results