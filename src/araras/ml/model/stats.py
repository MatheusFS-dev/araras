from araras.core import *

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import time
import warnings
from threading import Event, Lock, Thread

import psutil
import tensorflow as tf
import pynvml

from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from araras.ml.model.utils import DeviceKind, capture_model_summary, parse_device_spec
from araras.utils.misc import (
    format_number,
    format_bytes,
    format_scientific,
    format_number_commas,
)
from araras.utils.resource_monitor import ResourceMonitor
from araras.utils.system import format_metric_summary_line

MetricStatistic = Dict[str, Union[List[Union[int, float]], Union[int, float], None]]
ResourceMetricPayload = Dict[str, Union[str, MetricStatistic]]
ResourceMetricValue = Union[str, ResourceMetricPayload]
ResourceMetrics = Dict[str, ResourceMetricValue]


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
    """

    return get_flops(model, batch_size) // 2


def get_memory_and_time(
    model: tf.keras.Model,
    batch_size: int = 1,
    device: str = "both/0",
    warmup_runs: int = 10,
    test_runs: int = 20,
    verbose: bool = True,
) -> Tuple[
    Union[ResourceMetrics, Dict[str, Union[ResourceMetrics, str]]],
    Union[float, Dict[str, Union[float, str]]],
]:
    """
    Measure the exclusive resource footprint and average inference time of a
    ``tf.keras.Model`` on GPU or CPU.

    Observations:
        Warmup runs exclude one-time initialization costs from your measurements. On GPU
        the very first inference will trigger things like driver wake-up, context setup,
        PTX→BIN compilation and power-state switching, and cache fills. By running a few
        warmup inferences you force all of that work to happen before timing, so your
        measured latencies reflect true steady-state performance rather than setup overhead.

        Under @tf.function the first call also traces and builds the execution graph,
        applies optimizations and allocates buffers. Those activities inflate both time
        and memory on the “cold” run. Warmup runs let TensorFlow complete tracing and
        graph compilation once, so your timed loop measures only the optimized graph
        execution path.

    Notes:
        - The Keras Functional API may emit a ``UserWarning`` when the provided
          input structure does not exactly match the model's expected structure.
          This function suppresses that warning to keep console output tidy.
        - Model outputs may be nested structures (e.g., dict or list). Each
          tensor is converted to a NumPy array to synchronize execution across
          devices.

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        batch_size (int): The batch size to simulate for input. Defaults to 1.
            Measure with ``batch_size=1`` to get base per-sample latency.
        device (str): Canonical device selector. Accepts ``"cpu"`` for CPU-only
            profiling, ``"gpu/<index>"`` for a specific GPU, or ``"both/<index>"``
            to measure the CPU alongside GPU ``<index>``. Defaults to ``"both/0"``.
        warmup_runs (int): Number of warm-up runs before timing. Defaults to ``10``.
        test_runs (int): Number of runs to measure average inference time. Defaults
            to ``50``.
        verbose (bool): If ``True``, displays a progress bar during test runs.

    Raises:
        RuntimeError: If the specified GPU or CPU device cannot be found.
        ValueError: If ``device`` is not ``"cpu"``, ``"gpu/<index>"``, or ``"both/<index>"``.

    Returns:
        Tuple[Union[ResourceMetrics, Dict[str, Union[ResourceMetrics, str]]], Union[float, Dict[str, Union[float, str]]]]:
            - When a single device is measured, returns a ``ResourceMetrics`` mapping
              ``ram_used_bytes``, ``gpu_mem_used_bytes``, ``gpu_util_percent`` and
              ``cpu_util_percent`` to their respective statistics. When ``device``
              requests both CPU and GPU, a dictionary keyed by ``"gpu"`` and
              ``"cpu"`` is returned. Each value is either a ``ResourceMetrics``
              instance or ``"Error"`` if the measurement failed.
            - Average inference time in seconds for the measured device. When both
              devices are measured, returns a dictionary keyed by ``"gpu"`` and
              ``"cpu"`` with float values or ``"Error"`` if the run failed.
    """
    # Prepare dummy inputs matching model.inputs
    shapes = [(batch_size,) + tuple(K.int_shape(inp)[1:]) for inp in model.inputs]
    dummy_inputs = [tf.zeros(shape, dtype=inp.dtype) for shape, inp in zip(shapes, model.inputs)]

    @tf.function
    def infer(*args):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The structure of `inputs`",
                category=UserWarning,
            )
            return model(list(args), training=False)

    device_kind: DeviceKind
    resolved_gpu_index: Optional[int]
    device_kind, resolved_gpu_index = parse_device_spec(device)

    byte_metrics = {"ram_used_bytes", "gpu_mem_used_bytes"}

    def _empty_stats() -> MetricStatistic:
        return {
            "measurements": [],
            "min": None,
            "max": None,
            "avg": None,
            "std": None,
            "var": None,
        }

    def _build_stats(metric_name: str, value: Optional[float]) -> MetricStatistic:
        if value is None:
            return _empty_stats()
        cast_value: Union[int, float]
        if metric_name in byte_metrics:
            cast_value = int(round(value))
        else:
            cast_value = float(value)
        return {
            "measurements": [cast_value],
            "min": cast_value,
            "max": cast_value,
            "avg": cast_value,
            "std": 0.0,
            "var": 0.0,
        }

    def _summarize_monitor(
        monitor: ResourceMonitor, results: Dict[str, Optional[float]]
    ) -> Dict[str, ResourceMetricPayload]:
        summary: Dict[str, ResourceMetricPayload] = {}
        before_peaks = monitor.last_before_peaks
        during_peaks = monitor.last_during_peaks
        for metric_name, aggregation in monitor.metrics.items():
            before_value = before_peaks.get(metric_name)
            during_value = during_peaks.get(metric_name)
            aggregated_value = results.get(metric_name)
            summary[metric_name] = {
                "aggregation": aggregation,
                "before": _build_stats(metric_name, before_value),
                "during": _build_stats(metric_name, during_value),
                "delta": _build_stats(metric_name, aggregated_value),
            }
        return summary

    def _measure_gpu(gpu_index: int) -> Tuple[ResourceMetrics, float]:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus or gpu_index >= len(gpus):
            raise RuntimeError(f"No GPU found for index {gpu_index}")

        device_str = f"/GPU:{gpu_index}"
        tf.config.experimental.reset_memory_stats(device_str)

        _ = infer(*dummy_inputs)
        for _ in range(warmup_runs - 1):
            _ = infer(*dummy_inputs)

        monitor = ResourceMonitor(
            metrics={
                "ram_used_bytes": "delta",
                "ram_util_percent": "delta",
                "gpu_mem_used_bytes": "delta",
                "gpu_util_percent": "delta",
                "gpu_power_w": "peak",
                "cpu_util_percent": "delta",
                "cpu_power_rapl_w": "peak",
            },
            before_repetitions=3,
            during_repetitions=1,
            sample_interval_s=0.25,
            gpu_index=gpu_index,
            verbose=verbose,
        )

        times: List[float] = []

        def _workload() -> None:
            iterator = (
                white_track(
                    range(test_runs),
                    description=f"Calculating latency on GPU:{gpu_index}",
                    total=test_runs,
                )
                if verbose
                else range(test_runs)
            )
            for _ in iterator:
                t0 = time.perf_counter()
                out = infer(*dummy_inputs)
                tf.nest.map_structure(lambda t: t.numpy(), out)
                times.append(time.perf_counter() - t0)

        results = monitor.run_and_measure(_workload)
        avg_time = sum(times) / len(times) if times else 0.0
        return _summarize_monitor(monitor, results), avg_time

    def _measure_cpu() -> Tuple[ResourceMetrics, float]:
        if not tf.config.list_physical_devices("CPU"):
            raise RuntimeError("No CPU device found")

        cpu_index = 0
        monitor = ResourceMonitor(
            metrics={
                "ram_used_bytes": "delta",
                "ram_util_percent": "delta",
                "cpu_util_percent": "delta",
                "cpu_power_rapl_w": "peak",
            },
            before_repetitions=3,
            during_repetitions=1,
            sample_interval_s=0.25,
            gpu_index=resolved_gpu_index or 0,
            verbose=verbose,
        )

        with tf.device(f"/CPU:{cpu_index}"):
            _ = infer(*dummy_inputs)
            for _ in range(warmup_runs - 1):
                _ = infer(*dummy_inputs)

        times: List[float] = []

        def _workload() -> None:
            iterator = (
                white_track(
                    range(test_runs),
                    description="Profiling inference latency on CPU:0",
                    total=test_runs,
                )
                if verbose
                else range(test_runs)
            )
            for _ in iterator:
                with tf.device(f"/CPU:{cpu_index}"):
                    t0 = time.perf_counter()
                    out = infer(*dummy_inputs)
                    tf.nest.map_structure(lambda t: t.numpy(), out)
                times.append(time.perf_counter() - t0)

        results = monitor.run_and_measure(_workload)
        avg_time = sum(times) / len(times) if times else 0.0
        return _summarize_monitor(monitor, results), avg_time

    def _safe_measure(
        label: str,
        fn: Callable[[], Tuple[ResourceMetrics, float]],
    ) -> Tuple[Union[ResourceMetrics, str], Union[float, str]]:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive safeguard
            error_text = f"Error ({exc.__class__.__name__}): {exc}"
            if verbose:
                logger_error.error(f"Failed to measure {label.upper()} inference: {exc}")
            return error_text, error_text

    if device_kind == "gpu":
        if resolved_gpu_index is None:
            raise RuntimeError("GPU device index could not be resolved")
        return _safe_measure("gpu", lambda: _measure_gpu(resolved_gpu_index))

    if device_kind == "cpu":
        return _safe_measure("cpu", _measure_cpu)

    if resolved_gpu_index is None:
        raise RuntimeError("GPU device index could not be resolved")

    gpu_metrics, gpu_time = _safe_measure("gpu", lambda: _measure_gpu(resolved_gpu_index))
    cpu_metrics, cpu_time = _safe_measure("cpu", _measure_cpu)

    return {"gpu": gpu_metrics, "cpu": cpu_metrics}, {"gpu": gpu_time, "cpu": cpu_time}


def get_model_usage_stats(
    saved_model: str | tf.keras.Model,
    n_trials: int = 10000,
    device: str = "both/0",
    rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    verbose: bool = True,
) -> Union[
    Tuple[Union[float, str], Union[float, str], Union[float, str]],
    Dict[str, Dict[str, Union[float, str]]],
]:
    """
    Estimate average power draw and energy usage.
    Careful with the RAPL path; it may vary by system.
    The RAPL interface is typically found at:
        $ ls /sys/class/powercap
        intel-rapl
        $ ls /sys/class/powercap/intel-rapl
        intel-rapl:0       intel-rapl:0:0    intel-rapl:1    …
        $ ls /sys/class/powercap/intel-rapl/intel-rapl:0
        energy_uj  max_energy_range_uj  name
    Also, you MUST run this on a linux system with Intel CPUs!!!!!
    And run the python script with SUDO to access RAPL files.

    Notes:
        When a ``tf.keras.Model`` is provided, Keras may issue a ``UserWarning``
        about mismatched input structure if the dummy inputs do not precisely
        reflect the model's expected format. The warning is suppressed within
        this function.

    Args:
        saved_model (str | tf.keras.Model): Path to the TensorFlow SavedModel directory,
            a .keras model file, or a Keras Model instance.
        n_trials (int): Number of inference trials to perform. Defaults to ``10000``.
        device (str): Canonical device string. Use ``"cpu"`` for CPU-only
            profiling, ``"gpu/<index>"`` for a specific GPU, or ``"both/<index>"``
            to measure CPU usage alongside GPU ``<index>``. Defaults to
            ``"both/0"``.
        rapl_path (str): Path to the RAPL energy counter file for CPU measurements.
        verbose (bool): If ``True``, displays a progress bar during the trials.

    Raises:
        RuntimeError: If GPU NVML initialization fails when ``device`` refers to a GPU index.
        ValueError: If ``device`` is not ``"cpu"``, ``"gpu/<index>"``, or ``"both/<index>"``.

    Returns:
        Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], Dict[str, Dict[str, Union[float, str]]]]:
            - When a single device is measured, returns ``(per_run_time, avg_power, avg_energy)``.
            - When both CPU and GPU are requested, returns a dictionary keyed by
              ``"gpu"`` and ``"cpu"``. Each entry contains ``per_run_time``,
              ``avg_power`` and ``avg_energy``. On failure the corresponding values
              are set to ``"Error"``.
    """
    # Decide how we will run inference
    is_keras_model = False
    keras_model: Optional[tf.keras.Model] = None
    if isinstance(saved_model, tf.keras.Model) or (
        isinstance(saved_model, str) and saved_model.endswith(".keras")
    ):
        # We were given a Keras model or a path to a `.keras` file
        keras_model = (
            saved_model
            if isinstance(saved_model, tf.keras.Model)
            else tf.keras.models.load_model(saved_model)
        )
        is_keras_model = True

    # NVML will be initialized lazily inside the measurement loop when needed

    def read_cpu_power_rapl() -> Optional[float]:
        """Read CPU package energy via Intel RAPL interface.

        Opens the RAPL energy counter file and returns energy in joules.
        Returns None if the interface is unavailable or unreadable.
        """
        try:
            with open(rapl_path, "r") as f:
                # energy in microjoules; convert to joules
                return int(f.read()) / 1e6
        except Exception:
            # RAPL interface not present or unreadable
            raise RuntimeError(
                f"Unable to read CPU power from {rapl_path}. Ensure the path is correct and accessible, or run with sudo."
            )

    dummy_inputs = None
    infer: Callable[[], tf.Tensor]

    if is_keras_model:
        # Build random tensors based on the keras model input shapes
        tensors = []
        for tensor in keras_model.inputs:
            shape = [d if d is not None else 1 for d in tensor.shape]
            tensors.append(tf.random.normal(shape, dtype=tensor.dtype))
        dummy_inputs = tensors[0] if len(tensors) == 1 else tensors

        def infer():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The structure of `inputs`",
                    category=UserWarning,
                )
                return keras_model(dummy_inputs, training=False)

    else:
        # Load the SavedModel and obtain the serving_default signature for inference
        reloaded_model: tf.Module = tf.saved_model.load(saved_model)
        signature = reloaded_model.signatures["serving_default"]

        _, kwargs = signature.structured_input_signature
        inputs = {}
        for name, spec in kwargs.items():
            shape = [d if d is not None else 1 for d in spec.shape.as_list()]
            inputs[name] = tf.random.normal(shape, dtype=spec.dtype)
        dummy_inputs = inputs

        def infer():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The structure of `inputs`",
                    category=UserWarning,
                )
                return signature(**dummy_inputs)

    # Print the shapes of the input tensors
    # print(f"Input tensor shapes: {[t.shape for t in dummy_inputs.values()]}")

    device_kind: DeviceKind
    resolved_gpu_index: Optional[int]
    device_kind, resolved_gpu_index = parse_device_spec(device)

    def _measure_for_device(device_index: int) -> Tuple[float, float, float]:
        MAX_RETRIES = 2
        attempt = 0
        while True:
            powers: List[float] = []
            times: List[float] = []
            nvml_initialized = False
            handle = None

            try:
                if device_index >= 0:
                    pynvml.nvmlInit()
                    nvml_initialized = True
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as exc:
                if nvml_initialized:
                    pynvml.nvmlShutdown()
                raise RuntimeError(
                    "Unable to initialize NVML for GPU power monitoring: " + str(exc)
                ) from exc

            try:
                progress_iter = range(n_trials)
                if verbose:
                    device_label = (
                        f"GPU:{device_index}" if device_index >= 0 else "CPU:0"
                    )
                    progress_iter = white_track(
                        progress_iter,
                        description=f"Collecting stats on {device_label}",
                        total=n_trials,
                    )

                for _ in progress_iter:
                    start_time = time.time()
                    start_energy: Optional[float] = None

                    if device_index >= 0:
                        if handle is None:
                            raise RuntimeError("NVML handle not initialized")
                        start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    elif device_index == -1:
                        start_energy = read_cpu_power_rapl()
                    else:  # pragma: no cover - defensive guard
                        raise ValueError("Unsupported device index")

                    _ = infer()

                    elapsed = time.time() - start_time

                    if device_index >= 0:
                        if handle is None:
                            raise RuntimeError("NVML handle not initialized")
                        end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        avg_instant_power = (start_power + end_power) / 2
                        powers.append(avg_instant_power)
                    elif device_index == -1 and start_energy is not None:
                        end_energy = read_cpu_power_rapl()
                        if end_energy is not None:
                            energy_used = end_energy - start_energy
                            avg_power = energy_used / elapsed if elapsed > 0 else 0
                            powers.append(avg_power)

                    times.append(elapsed)
            finally:
                if device_index >= 0 and nvml_initialized:
                    pynvml.nvmlShutdown()

            per_run_time = sum(times) / len(times) if times else 0.0
            avg_power = sum(powers) / len(powers) if powers else 0.0
            avg_energy = (
                sum(p * t for p, t in zip(powers, times)) / len(powers) if powers else 0.0
            )

            if avg_power >= 0:
                return per_run_time, avg_power, avg_energy

            attempt += 1
            if attempt > MAX_RETRIES:
                logger_error.error(
                    f"{RED}Average power measurement failed after {MAX_RETRIES} attempts, returning 0.{RESET}"
                )
                return per_run_time, 0.0, 0.0

            logger.warning(
                f"{YELLOW}Negative average power measured, retrying measurement...{RESET}"
            )

    def _safe_measure(
        label: str,
        fn: Callable[[], Tuple[float, float, float]],
    ) -> Tuple[Union[float, str], Union[float, str], Union[float, str]]:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive safeguard
            error_text = f"Error ({exc.__class__.__name__}): {exc}"
            if verbose:
                logger_error.error(f"Failed to measure {label.upper()} usage: {exc}")
            return error_text, error_text, error_text

    if device_kind == "gpu":
        if resolved_gpu_index is None:
            raise RuntimeError("GPU device index could not be resolved")
        return _safe_measure("gpu", lambda: _measure_for_device(resolved_gpu_index))

    if device_kind == "cpu":
        return _safe_measure("cpu", lambda: _measure_for_device(-1))

    if resolved_gpu_index is None:
        raise RuntimeError("GPU device index could not be resolved")

    gpu_time, gpu_power, gpu_energy = _safe_measure(
        "gpu", lambda: _measure_for_device(resolved_gpu_index)
    )
    cpu_time, cpu_power, cpu_energy = _safe_measure("cpu", lambda: _measure_for_device(-1))

    return {
        "gpu": {
            "per_run_time": gpu_time,
            "avg_power": gpu_power,
            "avg_energy": gpu_energy,
        },
        "cpu": {
            "per_run_time": cpu_time,
            "avg_power": cpu_power,
            "avg_energy": cpu_energy,
        },
    }



def write_model_stats_to_file(
    model: tf.keras.Model,
    file_path: str,
    bytes_per_param: int,
    batch_size: int,
    device: str = "both/0",
    n_trials: int = 1000,
    extra_attrs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    stats_to_measure: Iterable[str] = (
        "parameters",
        "flops",
        "macs",
        "summary",
        "resource_usage",
        "usage_stats",
    ),
) -> None:
    """Write model statistics to a file.

    Statistics include:
        - Number of parameters
        - Model size in bytes
        - FLOPs (Floating Point Operations)
        - MACs (Multiply-Accumulate operations)
        - Resource usage deltas (system RAM, GPU RAM, GPU usage %, CPU usage %)
        - Inference time
        - Average power consumption
        - Average energy consumption

    Notes:
        The measured groups are controlled by ``stats_to_measure`` and any group
        not requested is reported as skipped in the generated report.

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        file_path (str): The path to the output file.
        bytes_per_param (int): Number of bytes per parameter for model size calculation.
        batch_size (int): The batch size to simulate for input.
        device (str): Canonical device selection string. Use ``"cpu"`` for CPU-only
            profiling, ``"gpu/<index>"`` for a specific GPU, or ``"both/<index>"`` to
            profile CPU and GPU sequentially (defaults to ``"both/0"``).
        n_trials (int): Number of trials for power and energy measurement.
        extra_attrs (Optional[Dict[str, Any]]): Additional attributes to write to the file.
        verbose (bool): If ``True``, print detailed information.
        stats_to_measure (Iterable[str]): Collection of statistic groups to compute.
            Supported values are ``"parameters"``, ``"flops"``, ``"macs"``,
            ``"summary"``, ``"resource_usage"`` and ``"usage_stats"``. Any omitted
            group is skipped.

    Returns:
        None: The report is written directly to ``file_path``.

    Raises:
        TypeError: If ``stats_to_measure`` is ``None`` or not iterable.
        ValueError: If ``stats_to_measure`` includes unsupported statistic names.
    """

    supported_stats: Tuple[str, ...] = (
        "parameters",
        "flops",
        "macs",
        "summary",
        "resource_usage",
        "usage_stats",
    )
    if stats_to_measure is None:
        raise TypeError("stats_to_measure must be an iterable of statistic names")

    try:
        stats_iterable = list(stats_to_measure)
    except TypeError as exc:
        raise TypeError("stats_to_measure must be an iterable of statistic names") from exc

    allowed_stats = set(supported_stats)
    selected_stats: List[str] = []
    invalid_entries: List[str] = []

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

    params_value: Union[int, str]
    model_size_value: Union[int, str]
    if measure_parameters:
        params_value = model.count_params()
        model_size_value = params_value * bytes_per_param
    else:
        params_value = skipped_text
        model_size_value = skipped_text

    flops_value: Union[int, str]
    if measure_flops:
        flops_value = get_flops(model)
    else:
        flops_value = skipped_text

    macs_value: Union[int, str]
    if measure_macs:
        macs_value = get_macs(model)
    else:
        macs_value = skipped_text

    model_summary_value = capture_model_summary(model) if measure_summary else skipped_text

    device_kind, _ = parse_device_spec(device)

    def _normalize_resource_usage(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict) and all(key in {"cpu", "gpu"} for key in raw.keys()):
            return {key: raw[key] for key in ("gpu", "cpu") if key in raw}
        target = "cpu" if device_kind == "cpu" else "gpu"
        return {target: raw}

    def _normalize_inference_time(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict) and all(key in {"cpu", "gpu"} for key in raw.keys()):
            return {key: raw[key] for key in ("gpu", "cpu") if key in raw}
        target = "cpu" if device_kind == "cpu" else "gpu"
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
        target = "cpu" if device_kind == "cpu" else "gpu"
        normalized[target] = {
            "per_run_time": per_run_time_value,
            "avg_power": avg_power_value,
            "avg_energy": avg_energy_value,
        }
        return normalized

    per_device_resource_usage: Dict[str, Any] = {}
    per_device_inference_time: Dict[str, Any] = {}
    per_device_usage_stats: Dict[str, Dict[str, Any]] = {}
    resource_usage_diff_map: Dict[str, Any] = {}
    resource_usage_display_map: Dict[str, Any] = {}

    resource_usage_primary: Any = not_measured
    resource_usage_diff_primary: Any = not_measured
    resource_usage_display_primary: Any = not_measured
    resource_usage_gpu: Any = not_measured
    resource_usage_cpu: Any = not_measured
    resource_usage_diff_gpu: Any = not_measured
    resource_usage_diff_cpu: Any = not_measured
    resource_usage_display_gpu: Any = not_measured
    resource_usage_display_cpu: Any = not_measured

    inference_time_primary: Any = not_measured
    inference_time_gpu: Any = not_measured
    inference_time_cpu: Any = not_measured

    avg_power_map: Dict[str, Any] = {}
    avg_energy_map: Dict[str, Any] = {}
    per_run_time_map: Dict[str, Any] = {}
    avg_power_primary: Any = not_measured
    avg_energy_primary: Any = not_measured
    per_run_time_primary: Any = not_measured
    avg_power_gpu: Any = not_measured
    avg_power_cpu: Any = not_measured
    avg_energy_gpu: Any = not_measured
    avg_energy_cpu: Any = not_measured
    per_run_time_gpu: Any = not_measured
    per_run_time_cpu: Any = not_measured

    default_resource_text = not_measured
    default_inference_text = not_measured
    default_usage_text = not_measured

    ram_metrics = {"ram_used_bytes", "gpu_mem_used_bytes"}

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

            aggregation_kind = (
                metric_value.get("aggregation")
                if isinstance(metric_value, dict)
                else "delta"
            )
            delta_stats = metric_value.get("delta") if isinstance(metric_value, dict) else None
            delta_max = None if not isinstance(delta_stats, dict) else delta_stats.get("max")
            before_stats = metric_value.get("before") if isinstance(metric_value, dict) else None
            during_stats = metric_value.get("during") if isinstance(metric_value, dict) else None
            during_max = None if not isinstance(during_stats, dict) else during_stats.get("max")

            if aggregation_kind == "peak":
                diff_value = during_max
            else:
                diff_value = delta_max
            diff_payload[metric_name] = (
                diff_value if diff_value is not None else not_measured
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
                        component_display[component_name] = (
                            f"{int(round(component_max))} B ({format_bytes(component_max)})"
                        )
                display_payload[metric_name] = component_display

        return diff_payload, display_payload

    if device_kind == "cpu":
        primary_order: Tuple[str, ...] = ("cpu", "gpu")
    elif device_kind == "gpu":
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

    def _value_for(container: Dict[str, Any], key: str, default: Any) -> Any:
        return container.get(key, default)

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

        resource_usage_primary = _pick_primary(per_device_resource_usage)
        resource_usage_diff_primary = _pick_primary(resource_usage_diff_map)
        resource_usage_display_primary = (
            _pick_primary(resource_usage_display_map) if resource_usage_display_map else not_measured
        )
        inference_time_primary = _pick_primary(per_device_inference_time)
        inference_time_gpu = _value_for(per_device_inference_time, "gpu", not_measured)
        inference_time_cpu = _value_for(per_device_inference_time, "cpu", not_measured)

        resource_usage_gpu = _value_for(per_device_resource_usage, "gpu", not_measured)
        resource_usage_cpu = _value_for(per_device_resource_usage, "cpu", not_measured)
        resource_usage_diff_gpu = _value_for(resource_usage_diff_map, "gpu", not_measured)
        resource_usage_diff_cpu = _value_for(resource_usage_diff_map, "cpu", not_measured)
        resource_usage_display_gpu = _value_for(
            resource_usage_display_map, "gpu", not_measured
        )
        resource_usage_display_cpu = _value_for(
            resource_usage_display_map, "cpu", not_measured
        )
    else:
        resource_usage_primary = skipped_text
        resource_usage_diff_primary = skipped_text
        resource_usage_display_primary = skipped_text
        inference_time_primary = skipped_text
        inference_time_gpu = skipped_text
        inference_time_cpu = skipped_text
        resource_usage_gpu = skipped_text
        resource_usage_cpu = skipped_text
        resource_usage_diff_gpu = skipped_text
        resource_usage_diff_cpu = skipped_text
        resource_usage_display_gpu = skipped_text
        resource_usage_display_cpu = skipped_text
        default_resource_text = skipped_text
        default_inference_text = skipped_text

    if measure_usage_stats:
        usage_stats_raw = get_model_usage_stats(
            model, device=device, n_trials=n_trials, verbose=verbose
        )
        per_device_usage_stats = _normalize_usage_stats(usage_stats_raw)
        for device_label, stats_payload in per_device_usage_stats.items():
            avg_power_map[device_label] = stats_payload.get("avg_power", not_measured)
            avg_energy_map[device_label] = stats_payload.get("avg_energy", not_measured)
            per_run_time_map[device_label] = stats_payload.get("per_run_time", not_measured)

        avg_power_primary = _pick_primary(avg_power_map)
        avg_energy_primary = _pick_primary(avg_energy_map)
        per_run_time_primary = _pick_primary(per_run_time_map)

        avg_power_gpu = _value_for(avg_power_map, "gpu", not_measured)
        avg_power_cpu = _value_for(avg_power_map, "cpu", not_measured)
        avg_energy_gpu = _value_for(avg_energy_map, "gpu", not_measured)
        avg_energy_cpu = _value_for(avg_energy_map, "cpu", not_measured)
        per_run_time_gpu = _value_for(per_run_time_map, "gpu", not_measured)
        per_run_time_cpu = _value_for(per_run_time_map, "cpu", not_measured)
    else:
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

    model_stats = {
        "num_params": params_value,
        "model_size": model_size_value,
        "flops": flops_value,
        "macs": macs_value,
        "model_summary": model_summary_value,
        "resource_usage": resource_usage_primary,
        "resource_usage_diff": resource_usage_diff_primary,
        "resource_usage_display": resource_usage_display_primary,
        "resource_usage_details": per_device_resource_usage,
        "resource_usage_diff_details": resource_usage_diff_map,
        "resource_usage_display_details": resource_usage_display_map,
        "resource_usage_gpu": resource_usage_gpu,
        "resource_usage_cpu": resource_usage_cpu,
        "resource_usage_diff_gpu": resource_usage_diff_gpu,
        "resource_usage_diff_cpu": resource_usage_diff_cpu,
        "resource_usage_display_gpu": resource_usage_display_gpu,
        "resource_usage_display_cpu": resource_usage_display_cpu,
        "inference_time": inference_time_primary,
        "inference_time_details": per_device_inference_time,
        "inference_time_gpu": inference_time_gpu,
        "inference_time_cpu": inference_time_cpu,
        "avg_power": avg_power_primary,
        "avg_power_details": avg_power_map,
        "avg_power_gpu": avg_power_gpu,
        "avg_power_cpu": avg_power_cpu,
        "avg_energy": avg_energy_primary,
        "avg_energy_details": avg_energy_map,
        "avg_energy_gpu": avg_energy_gpu,
        "avg_energy_cpu": avg_energy_cpu,
        "per_run_time": per_run_time_primary,
        "per_run_time_details": per_run_time_map,
        "per_run_time_gpu": per_run_time_gpu,
        "per_run_time_cpu": per_run_time_cpu,
    }

    extra_attrs = extra_attrs or {}

    with open(file_path, "w") as file:
        num_params_entry = model_stats["num_params"]
        if isinstance(num_params_entry, str):
            file.write(f"Number of parameters: {num_params_entry}\n")
        else:
            file.write(f"Number of parameters: {format_number_commas(num_params_entry)}\n")

        model_size_entry = model_stats["model_size"]
        if isinstance(model_size_entry, str):
            file.write(f"Model size: {model_size_entry}\n")
        else:
            file.write(f"Model size: {format_bytes(model_size_entry)}\n")

        flops_entry = model_stats["flops"]
        if isinstance(flops_entry, str):
            file.write(f"FLOPs: {flops_entry}\n")
        else:
            file.write(f"FLOPs: {format_number(flops_entry)}FLOPs\n")

        macs_entry = model_stats["macs"]
        if isinstance(macs_entry, str):
            file.write(f"MACs: {macs_entry}\n")
        else:
            file.write(f"MACs: {format_number(macs_entry)}MACs\n")

        device_keys = list(per_device_resource_usage.keys())
        if not device_keys:
            device_keys = ["overall"]

        ordered_devices: List[str] = []
        for candidate in ("gpu", "cpu"):
            if candidate in per_device_resource_usage and candidate not in ordered_devices:
                ordered_devices.append(candidate)
        for key in device_keys:
            if key not in ordered_devices:
                ordered_devices.append(key)

        optional_devices: Set[str] = set()
        if "gpu" not in ordered_devices:
            ordered_devices.insert(0, "gpu")
            optional_devices.add("gpu")
        if "cpu" not in ordered_devices:
            ordered_devices.append("cpu")
            optional_devices.add("cpu")

        actual_measured = [
            label for label in ordered_devices if label not in optional_devices and label != "overall"
        ]
        multiple_devices = len(actual_measured) > 1

        def _format_failure(device_label: str, label: str, reason: str | None) -> str:
            normalized_reason = None if reason is None else str(reason).strip().lower()
            if device_label in optional_devices:
                if reason is None or normalized_reason == "not measured":
                    return f"{label}: Not measured"
                if normalized_reason and normalized_reason.startswith("not measured (skipped"):
                    return f"{label}: {reason}"

            text = "Not measured" if reason is None else str(reason)
            lowered = text.strip().lower()
            if lowered == "not measured" and multiple_devices:
                text = "Error: Measurement unavailable"
            elif not lowered.startswith("error"):
                text = f"Error: {text}"
            if text.lower().startswith("error"):
                logger_error.error(
                    f"{label} measurement failed for {device_label.upper()}: {text}"
                )
            return f"{label}: {text}"

        def _metric_line(
            device_label: str,
            metric_key: str,
            label: str,
            *,
            is_ram: bool = False,
        ) -> Optional[str]:
            metrics_value = per_device_resource_usage.get(device_label, default_resource_text)
            if isinstance(metrics_value, str):
                lowered = metrics_value.strip().lower()
                if lowered.startswith("not measured (skipped"):
                    return f"{label}: {metrics_value}"
                if lowered == "not measured" and (
                    device_label in optional_devices or not multiple_devices
                ):
                    return f"{label}: Not measured"
                return _format_failure(device_label, label, metrics_value)
            if not isinstance(metrics_value, dict):
                return None

            metric_payload = metrics_value.get(metric_key)
            if metric_payload is None:
                return None
            if isinstance(metric_payload, str):
                lowered = metric_payload.strip().lower()
                if lowered.startswith("not measured (skipped"):
                    return f"{label}: {metric_payload}"
                if lowered == "not measured" and (
                    device_label in optional_devices or not multiple_devices
                ):
                    return f"{label}: Not measured"
                return _format_failure(device_label, label, metric_payload)

            before_stats = metric_payload.get("before") if isinstance(metric_payload, dict) else None
            during_stats = metric_payload.get("during") if isinstance(metric_payload, dict) else None
            delta_stats = metric_payload.get("delta") if isinstance(metric_payload, dict) else None

            before_value = None if not isinstance(before_stats, dict) else before_stats.get("max")
            during_value = None if not isinstance(during_stats, dict) else during_stats.get("max")
            diff_value = None if not isinstance(delta_stats, dict) else delta_stats.get("max")
            return format_metric_summary_line(
                label,
                before_value,
                during_value,
                diff_value,
                is_byte_metric=is_ram,
            )

        def _scalar_line(
            device_label: str,
            label: str,
            value: Union[float, str, None],
            *,
            unit: Optional[str] = None,
        ) -> Optional[str]:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered.startswith("not measured (skipped"):
                    return f"{label}: {value}"
                if lowered == "not measured" and (
                    device_label in optional_devices or not multiple_devices
                ):
                    return f"{label}: Not measured"
                return _format_failure(device_label, label, value)
            if value is None:
                if multiple_devices and device_label not in optional_devices:
                    return _format_failure(device_label, label, None)
                return f"{label}: Not measured"

            formatted = format_scientific(value, max_precision=4)
            if unit:
                formatted = f"{formatted} {unit}"
            return f"{label}: {formatted}"

        for index, device_label in enumerate(ordered_devices):
            device_name = {"gpu": "GPU", "cpu": "CPU"}.get(device_label, device_label.upper())
            file.write(f"{device_name} Inference:\n")

            system_line = _metric_line(device_label, "ram_used_bytes", "System Memory", is_ram=True)
            if system_line:
                file.write(f"    - {system_line}\n")

            if device_label == "gpu":
                gpu_mem_line = _metric_line(device_label, "gpu_mem_used_bytes", "GPU Memory", is_ram=True)
                if gpu_mem_line:
                    file.write(f"    - {gpu_mem_line}\n")
                gpu_usage_line = _metric_line(device_label, "gpu_util_percent", "GPU Usage")
                if gpu_usage_line:
                    file.write(f"    - {gpu_usage_line}\n")
                gpu_diff_entry = (
                    diff_entry if isinstance(diff_entry, dict) else {}
                )
                gpu_power_line = _scalar_line(
                    device_label,
                    "GPU Power",
                    gpu_diff_entry.get("gpu_power_w", diff_entry)
                    if isinstance(diff_entry, dict)
                    else diff_entry,
                    unit="W",
                )
                if gpu_power_line:
                    file.write(f"    - {gpu_power_line}\n")
            elif device_label == "cpu":
                cpu_usage_line = _metric_line(device_label, "cpu_util_percent", "CPU Usage")
                if cpu_usage_line:
                    file.write(f"    - {cpu_usage_line}\n")
                cpu_diff_entry = (
                    diff_entry if isinstance(diff_entry, dict) else {}
                )
                cpu_power_line = _scalar_line(
                    device_label,
                    "CPU Power",
                    cpu_diff_entry.get("cpu_power_rapl_w", diff_entry)
                    if isinstance(diff_entry, dict)
                    else diff_entry,
                    unit="W",
                )
                if cpu_power_line:
                    file.write(f"    - {cpu_power_line}\n")
            else:
                gpu_mem_line = _metric_line(device_label, "gpu_mem_used_bytes", "GPU Memory", is_ram=True)
                if gpu_mem_line:
                    file.write(f"    - {gpu_mem_line}\n")
                gpu_usage_line = _metric_line(device_label, "gpu_util_percent", "GPU Usage")
                if gpu_usage_line:
                    file.write(f"    - {gpu_usage_line}\n")
                cpu_usage_line = _metric_line(device_label, "cpu_util_percent", "CPU Usage")
                if cpu_usage_line:
                    file.write(f"    - {cpu_usage_line}\n")
                both_diff_entry = (
                    diff_entry if isinstance(diff_entry, dict) else {}
                )
                gpu_power_line = _scalar_line(
                    device_label,
                    "GPU Power",
                    both_diff_entry.get("gpu_power_w", diff_entry)
                    if isinstance(diff_entry, dict)
                    else diff_entry,
                    unit="W",
                )
                if gpu_power_line:
                    file.write(f"    - {gpu_power_line}\n")
                cpu_power_line = _scalar_line(
                    device_label,
                    "CPU Power",
                    both_diff_entry.get("cpu_power_rapl_w", diff_entry)
                    if isinstance(diff_entry, dict)
                    else diff_entry,
                    unit="W",
                )
                if cpu_power_line:
                    file.write(f"    - {cpu_power_line}\n")

            inference_line = _scalar_line(
                device_label,
                "Inference Time",
                per_device_inference_time.get(device_label, default_inference_text),
                unit="s",
            )
            if inference_line:
                file.write(f"    - {inference_line}\n")

            power_line = _scalar_line(
                device_label,
                "Power Consumption",
                avg_power_map.get(device_label, default_usage_text),
                unit="W",
            )
            if power_line:
                file.write(f"    - {power_line}\n")

            energy_line = _scalar_line(
                device_label,
                "Energy Consumption",
                avg_energy_map.get(device_label, default_usage_text),
                unit="J",
            )
            if energy_line:
                file.write(f"    - {energy_line}\n")
            if index != len(ordered_devices) - 1:
                file.write("\n")

        for attr, value in extra_attrs.items():
            file.write(f"{attr}: {value}\n")

        file.write(f"\nModel summary: {model_stats['model_summary']}\n")
