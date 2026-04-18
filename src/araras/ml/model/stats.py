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
from araras.utils.misc import format_bytes, format_number, format_scientific
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

    logger = tf.get_logger()
    old_level = logger.level
    logger.setLevel("ERROR")
    try:

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
    finally:
        logger.setLevel(old_level)

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
        bytes_per_param (int): Number of bytes assigned to each trainable
            parameter when estimating the model size. Defaults to ``4`` (the
            footprint of ``float32`` weights).

    Returns:
        Dict[str, Any]: Mapping from metric names to their computed statistics.
        Latency metrics return a mapping with ``average_s`` and ``peak_s``.
        Resource metrics provide aggregation metadata, the collected
        measurements, and simple summary statistics. The ``model_size`` metric
        reports the estimated footprint in bytes. Metrics that cannot be
        computed return ``None`` or an error string.

    Raises:
        ValueError: If ``device`` resolves to ``"both"``, if ``test_runs`` is
            less than ``1``, or if ``bytes_per_param`` is less than ``1``.
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

    if bytes_per_param < 1:
        raise ValueError("bytes_per_param must be at least 1")

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
    results: Dict[str, Any] = {
        "batch_size": batch_size,
        "parameters_batch_size": batch_size,
    }

    parameter_cache: Optional[int] = None

    def _get_parameter_count() -> int:
        nonlocal parameter_cache
        if parameter_cache is None:
            parameter_cache = int(model.count_params())
        return parameter_cache

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
            results[stat] = _get_parameter_count()
        elif stat == "model_size":
            param_count = _get_parameter_count()
            results[stat] = int(param_count * bytes_per_param)
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
                f"Unknown statistic '{stat}' requested\nAccepted values: {list(RESOURCE_METRIC_AGGREGATIONS.keys()) + ['parameters', 'model_size', 'flops', 'macs', 'summary', 'inference_latency']}"
            )

    return results


def render_model_stats_report(
    structural_stats: Dict[str, Any],
    *,
    cpu_stats: Optional[Dict[str, Any]] = None,
    gpu_stats: Optional[Dict[str, Any]] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a textual report summarising collected model statistics.

    The helper accepts raw metric dictionaries produced by
    :func:`get_model_stats` and organises them into the same
    human-readable layout consumed by :func:`write_model_stats_to_file`.
    Engineering and scientific notation are used for improved readability
    while still retaining the original numeric values.

    Args:
        structural_stats (Dict[str, Any]): Baseline statistics such as parameter
            count, FLOPs, MACs, and model summary text. Typically this is one of
            the dictionaries returned by :func:`get_model_stats`.
        cpu_stats (Optional[Dict[str, Any]]): Statistics gathered on the CPU, if
            available. Defaults to ``None``.
        gpu_stats (Optional[Dict[str, Any]]): Statistics gathered on the GPU, if
            available. Defaults to ``None``.
        extra_attrs (Optional[Dict[str, Any]]): Additional attributes appended to
            the end of the report. Defaults to ``None``.

    Returns:
        str: Multiline textual summary ready to be written to a file or stored
        in trial metadata.

    Notes:
        Missing metrics render as ``"N/A"`` to highlight unavailable
        measurements without raising exceptions.

    Warnings:
        Supplying dictionaries that do not match the schema returned by
        :func:`get_model_stats` may result in incomplete or mislabelled
        sections in the generated text.
    """

    extra_attrs = extra_attrs or {}
    structural_stats = structural_stats or {}

    def _format_plain(value: Optional[Any]) -> str:
        return "N/A" if value is None else str(value)

    def _format_engineering(value: Optional[Any]) -> str:
        return "N/A" if value is None else format_number(value, precision=2)

    def _format_bytes(value: Optional[Any]) -> str:
        return "N/A" if value is None else format_bytes(value, precision=2)

    def _format_scientific(value: Optional[Any]) -> str:
        return "N/A" if value is None else format_scientific(value, max_precision=4)

    def _get_summary_metric(
        stats: Optional[Dict[str, Any]], key: str, field: str = "average"
    ) -> Optional[float]:
        if not stats:
            return None
        data = stats.get(key)
        if isinstance(data, dict):
            return data.get(field)
        return data  # type: ignore[return-value]

    def _get_latency(stats: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
        if not stats:
            return None, None
        data = stats.get("inference_latency")
        if isinstance(data, dict):
            return data.get("average_s"), data.get("peak_s")
        return None, None

    lines: List[str] = []

    parameter_count = structural_stats.get("parameters")
    parameter_label = "Number of Parameters"
    lines.append(
        f"{parameter_label}: {_format_plain(parameter_count)} parameters ({_format_engineering(parameter_count)})"
    )

    model_size_bytes = structural_stats.get("model_size")
    lines.append(f"Model Size: {_format_plain(model_size_bytes)} B ({_format_bytes(model_size_bytes)})")

    flops = structural_stats.get("flops")
    flops_batch_size = structural_stats.get("flops_batch_size")
    if flops_batch_size is None:
        flops_batch_size = structural_stats.get("batch_size")
    if flops_batch_size is None:
        flops_batch_size = structural_stats.get("parameters_batch_size")
    if flops_batch_size is None:
        flops_batch_size = (
            extra_attrs.get("flops_batch_size")
            or extra_attrs.get("batch_size")
            or extra_attrs.get("parameters_batch_size")
        )
    flops_label = "FLOPs"
    if flops_batch_size is not None:
        flops_label += f" (batch size: {flops_batch_size})"
    lines.append(f"{flops_label}: {_format_plain(flops)} FLOPs ({_format_engineering(flops)})")

    lines.append("")

    if gpu_stats:
        gpu_avg_latency, gpu_peak_latency = _get_latency(gpu_stats)
        gpu_system_memory = _get_summary_metric(gpu_stats, "ram_used_bytes")
        gpu_memory = _get_summary_metric(gpu_stats, "gpu_mem_used_bytes")
        gpu_usage = _get_summary_metric(gpu_stats, "gpu_util_percent")
        gpu_power_peak = _get_summary_metric(gpu_stats, "gpu_power_w", field="peak")
        gpu_power_avg = _get_summary_metric(gpu_stats, "gpu_power_w")
        gpu_energy = None
        if gpu_power_avg is not None and gpu_avg_latency is not None:
            gpu_energy = gpu_power_avg * gpu_avg_latency

        lines.append("GPU Inference:")
        lines.append(
            f"    - System Memory: {_format_plain(gpu_system_memory)} B ({_format_bytes(gpu_system_memory)})"
        )
        lines.append(f"    - GPU Memory: {_format_plain(gpu_memory)} B ({_format_bytes(gpu_memory)})")
        gpu_usage_str = "N/A" if gpu_usage is None else f"{gpu_usage:.2f} %"
        lines.append(f"    - GPU Usage: {gpu_usage_str}")
        lines.append(f"    - GPU Power: {_format_scientific(gpu_power_peak)} W")
        lines.append(
            "    - Inference Time (avg/peak): "
            f"{_format_scientific(gpu_avg_latency)} s / {_format_scientific(gpu_peak_latency)} s"
        )
        lines.append(f"    - Energy Consumption: {_format_scientific(gpu_energy)} J")

    if cpu_stats:
        cpu_avg_latency, cpu_peak_latency = _get_latency(cpu_stats)
        cpu_system_memory = _get_summary_metric(cpu_stats, "ram_used_bytes")
        cpu_usage_summary = cpu_stats.get("cpu_util_percent") if isinstance(cpu_stats, dict) else None
        cpu_power_peak = _get_summary_metric(cpu_stats, "cpu_power_rapl_w", field="peak")
        cpu_power_avg = _get_summary_metric(cpu_stats, "cpu_power_rapl_w")
        cpu_energy = None
        if cpu_power_avg is not None and cpu_avg_latency is not None:
            cpu_energy = cpu_power_avg * cpu_avg_latency

        usage_line = "N/A"
        if isinstance(cpu_usage_summary, dict):
            usage_max = cpu_usage_summary.get("max")
            usage_min = cpu_usage_summary.get("min")
            if usage_max is not None and usage_min is not None:
                usage_delta = usage_max - usage_min
                usage_line = f"{usage_delta:.2f}%"

        lines.append("CPU Inference:")
        lines.append(
            f"    - System Memory: {_format_plain(cpu_system_memory)} B ({_format_bytes(cpu_system_memory)})"
        )
        lines.append(f"    - CPU Usage: {usage_line}")
        lines.append(f"    - CPU Power: {_format_scientific(cpu_power_peak)} W")
        lines.append(
            "    - Inference Time (avg/peak): "
            f"{_format_scientific(cpu_avg_latency)} s / {_format_scientific(cpu_peak_latency)} s"
        )
        lines.append(f"    - Energy Consumption: {_format_scientific(cpu_energy)} J")

    lines.append("")

    summary_text = structural_stats.get("summary")
    lines.append("Model Summary:")
    if isinstance(summary_text, str) and summary_text.strip():
        lines.extend(summary_text.splitlines())
    else:
        lines.append("N/A")

    if extra_attrs:
        lines.append("")
        for key, value in extra_attrs.items():
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def write_model_stats_to_file(
    model: tf.keras.Model,
    file_path: str,
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
) -> None:
    """Persist model statistics obtained via :func:`get_model_stats` to a file.

    Args:
        model (tf.keras.Model): Model instance analysed for statistics.
        file_path (str): Destination path of the report file. Intermediate
            directories must already exist.
        batch_size (int): Batch size forwarded to :func:`get_model_stats`.
            Defaults to ``1``.
        device (str): Canonical device selector. Use ``"cpu"`` for CPU-only
            profiling, ``"gpu/<index>"`` for a specific GPU, or ``"both/<index>``
            to capture CPU and GPU statistics sequentially. Defaults to
            ``"both/0"``.
        stats_to_measure (Iterable[str]): Metric identifiers forwarded to
            :func:`get_model_stats`. Defaults to the standard metric set.
        test_runs (int): Number of repetitions for each resource metric. Defaults
            to ``10``.
        verbose (int): Verbosity forwarded to :func:`get_model_stats`. Defaults
            to ``1``.
        bytes_per_param (int): Number of bytes assigned to each model parameter
            when estimating the ``model_size`` metric. Defaults to ``4``.
        extra_attrs (Optional[Dict[str, Any]]): Additional attributes appended to
            the report after the model summary. Defaults to ``None``.

    Raises:
        ValueError: If ``device`` cannot be parsed or resolves to ``"both"``
            without an index, or if ``bytes_per_param`` is less than ``1``.
        RuntimeError: Propagated if the requested GPU index is unavailable.

    Notes:
        The function overwrites ``file_path`` if it already exists. Extra
        attributes are emitted in the order provided by ``extra_attrs``.

    Warnings:
        Missing sensors or monitoring permissions can lead to ``"N/A"`` values
        in the report because :func:`get_model_stats` silently skips metrics
        that cannot be collected.
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

    structural_stats = next((stats for stats in (gpu_stats, cpu_stats) if stats), {})
    report_text = render_model_stats_report(
        structural_stats,
        cpu_stats=cpu_stats,
        gpu_stats=gpu_stats,
        extra_attrs=extra_attrs,
    )

    with open(file_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text + "\n")
