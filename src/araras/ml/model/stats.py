from araras.core import *

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import time
import warnings
import tensorflow as tf
import pynvml

from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from araras.ml.model.utils import capture_model_summary
from araras.utils.misc import (
    format_number,
    format_bytes,
    format_scientific,
    format_number_commas,
)
from araras.utils.system import measure_system_resources

ResourceMetricValue = Union[str, Dict[str, Union[int, float]]]
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
    device: int = 0,
    warmup_runs: int = 10,
    test_runs: int = 50,
    verbose: bool = True,
) -> Tuple[ResourceMetrics, float]:
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
            Measure with batch_size=1 to get base per-sample latency.
        device (int): GPU index to run the model on. Use ``-1`` to run on CPU.
        warmup_runs (int): Number of warm-up runs before timing. Defaults to 10.
        test_runs (int): Number of runs to measure average inference time. Defaults to 50.
        verbose (bool): If True, displays a progress bar during test runs.

    Raises:
        RuntimeError: If the specified GPU or CPU device cannot be found.

    Returns:
        Tuple[ResourceMetrics, float]:
            - Dictionary with the average per-inference resource statistics. Keys are
              ``system_ram``, ``gpu_ram``, ``gpu_usage`` and ``cpu_usage``. For each
              key the value is either ``"Not measured"`` or a mapping containing the
              average ``before`` value, the ``current`` value and their ``difference``.
              RAM measurements are expressed in bytes; usage values are percentages.
            - Average inference time in seconds.
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

    use_gpu = device >= 0
    device_str = f"/GPU:{device}" if use_gpu else "/CPU:0"

    metric_keys = ("system_ram", "gpu_ram", "gpu_usage", "cpu_usage")
    ram_metrics = {"system_ram", "gpu_ram"}

    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _capture_snapshot() -> Dict[str, Optional[float]]:
        metrics = ["ram"]
        if use_gpu:
            metrics.append("gpu_ram")
        else:
            metrics.append("cpu")

        snapshot = {key: None for key in metric_keys}

        try:
            raw_results = measure_system_resources(",".join(metrics))
        except Exception:  # pragma: no cover - defensive safeguard
            return snapshot

        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            if entry.get("error"):
                continue

            metric_name = entry.get("metric")
            if metric_name == "ram":
                snapshot["system_ram"] = _safe_float(entry.get("used_bytes"))
            elif metric_name == "cpu":
                snapshot["cpu_usage"] = _safe_float(entry.get("percent"))
            elif metric_name == "gpu_ram":
                gpus = entry.get("gpus", [])
                for gpu in gpus:
                    if gpu.get("index") != device:
                        continue
                    used_mb = _safe_float(gpu.get("used_mb"))
                    if used_mb is not None:
                        snapshot["gpu_ram"] = used_mb * 1024 * 1024
                    snapshot["gpu_usage"] = _safe_float(gpu.get("utilization_percent"))
                    break

        return snapshot

    totals = {
        key: {"before": 0.0, "current": 0.0, "difference": 0.0}
        for key in metric_keys
    }
    counts = {key: 0 for key in metric_keys}

    def _update_metrics(
        before: Dict[str, Optional[float]],
        after: Dict[str, Optional[float]],
    ) -> None:
        for key in metric_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            if before_val is None or after_val is None:
                continue

            diff = after_val - before_val
            if diff < 0:
                diff = 0.0

            totals[key]["before"] += before_val
            totals[key]["current"] += after_val
            totals[key]["difference"] += diff
            counts[key] += 1

    def _cast_metric_value(key: str, value: float) -> Union[int, float]:
        if key in ram_metrics:
            return int(round(value))
        return value

    def _finalize_metrics() -> ResourceMetrics:
        final: ResourceMetrics = {}
        for key in metric_keys:
            count = counts[key]
            if count == 0:
                final[key] = "Not measured"
                continue

            final[key] = {
                "before": _cast_metric_value(key, totals[key]["before"] / count),
                "current": _cast_metric_value(key, totals[key]["current"] / count),
                "difference": _cast_metric_value(key, totals[key]["difference"] / count),
            }

        return final

    if use_gpu:
        # Verify GPU exists
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus or device >= len(gpus):
            raise RuntimeError(f"No GPU found for index {device}")

        # Reset tracked peak to include weights + graph
        tf.config.experimental.reset_memory_stats(device_str)

        # Warm-up runs (first includes trace + alloc)
        _ = infer(*dummy_inputs)
        for _ in range(warmup_runs - 1):
            _ = infer(*dummy_inputs)

        # Timed inference with forced sync
        times = []
        progress_iter = range(test_runs)
        if verbose:
            progress_iter = white_track(
                progress_iter,
                description="Measuring GPU",
                total=test_runs,
            )
        for _ in progress_iter:
            before_snapshot = _capture_snapshot()
            t0 = time.perf_counter()
            out = infer(*dummy_inputs)
            tf.nest.map_structure(lambda t: t.numpy(), out)
            times.append(time.perf_counter() - t0)
            after_snapshot = _capture_snapshot()
            _update_metrics(before_snapshot, after_snapshot)

        avg_time = sum(times) / len(times)
        resource_metrics = _finalize_metrics()
        return resource_metrics, avg_time

    # CPU path
    if not tf.config.list_physical_devices("CPU"):
        raise RuntimeError("No CPU device found")

    cpu_index = 0
    with tf.device(f"/CPU:{cpu_index}"):
        _ = infer(*dummy_inputs)
        for _ in range(warmup_runs - 1):
            _ = infer(*dummy_inputs)

    times = []
    progress_iter = range(test_runs)
    if verbose:
        progress_iter = white_track(
            progress_iter,
            description="Measuring CPU",
            total=test_runs,
        )

    for _ in progress_iter:
        before_snapshot = _capture_snapshot()
        with tf.device(f"/CPU:{cpu_index}"):
            t0 = time.perf_counter()
            out = infer(*dummy_inputs)
            tf.nest.map_structure(lambda t: t.numpy(), out)
        times.append(time.perf_counter() - t0)
        after_snapshot = _capture_snapshot()
        _update_metrics(before_snapshot, after_snapshot)

    avg_time = sum(times) / len(times)
    resource_metrics = _finalize_metrics()
    return resource_metrics, avg_time


def get_model_usage_stats(
    saved_model: str | tf.keras.Model,
    n_trials: int = 10000,
    device: int = 0,
    rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    verbose: bool = True,
) -> Tuple[float, float, float]:
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
        n_trials (int): Number of inference trials to perform. Defaults to 100000.
        device (int): GPU index for power measurement, or ``-1`` to use the CPU.
        rapl_path (str): Path to the RAPL energy counter file for CPU measurements.
        verbose (bool): If True, displays a progress bar during the trials.

    Raises:
        RuntimeError: If GPU NVML initialization fails when ``device`` refers to a GPU index.
        ValueError: If ``device`` is neither ``-1`` nor a valid GPU index.

    Returns:
        Tuple[float, float, float]:
            - per_run_time (float):
                Average run time in seconds. Measures a mix of tracing, initialization,
                asynchronous queuing, Python overhead, and power-reading delays,
                so its “average” can be dominated by non-inference costs.
            - avg_power (float): Average power draw in watts. If a negative value is
              measured repeatedly, the function returns 0 after two retries.
            - avg_energy (float): Average energy consumed per inference in joules. This
              will also be ``0`` if ``avg_power`` could not be measured correctly.
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
                f"\033[91mUnable to read CPU power from {rapl_path}. "
                "Ensure the path is correct and accessible, or run with sudo.\033[0m"
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

    MAX_RETRIES = 2
    attempt = 0
    while True:
        powers: list[float] = []  # store measured power values
        times: list[float] = []  # store inference durations

        # Initialize NVML each attempt if required
        if device >= 0:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            except Exception as e:
                raise RuntimeError("Unable to initialize NVML for GPU power monitoring: " + str(e))

        progress_iter = range(n_trials)
        if verbose:
            progress_iter = white_track(
                progress_iter,
                description="Measuring usage",
                total=n_trials,
            )
        for _ in progress_iter:

            start_time = time.time()

            if device >= 0:
                start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            elif device == -1:
                start_energy = read_cpu_power_rapl()
            else:
                raise ValueError("Unsupported device index")

            _ = infer()

            elapsed = time.time() - start_time

            if device >= 0:
                end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                avg_instant_power = (start_power + end_power) / 2
                powers.append(avg_instant_power)
            elif device == -1 and start_energy is not None:
                end_energy = read_cpu_power_rapl()
                if end_energy is not None:
                    energy_used = end_energy - start_energy
                    avg_power = energy_used / elapsed if elapsed > 0 else 0
                    powers.append(avg_power)

            times.append(elapsed)

        if device >= 0:
            pynvml.nvmlShutdown()

        per_run_time = sum(times) / len(times)
        avg_power = sum(powers) / len(powers) if powers else 0
        avg_energy = sum(p * t for p, t in zip(powers, times)) / len(powers) if powers else 0

        if avg_power >= 0:
            return per_run_time, avg_power, avg_energy

        attempt += 1
        if attempt > MAX_RETRIES:
            logger_error.error(
                f"{RED}Average power measurement failed after {MAX_RETRIES} attempts, returning 0.{RESET}"
            )

            return per_run_time, 0.0, 0.0
        logger.warning(f"{YELLOW}Negative average power measured, retrying measurement...{RESET}")


def write_model_stats_to_file(
    model: tf.keras.Model,
    file_path: str,
    bytes_per_param: int,
    batch_size: int,
    device: int = 0,
    n_trials: int = 1000,
    extra_attrs: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """
    Write model statistics to a file.
    
    Statistics include:
        - Number of parameters
        - Model size in bits
        - FLOPs (Floating Point Operations)
        - MACs (Multiply-Accumulate operations)
        - Average per-inference resource deltas (system RAM, GPU RAM, GPU usage %, CPU usage %)
        - Inference time
        - Average power consumption
        - Average energy consumption

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        file_path (str): The path to the output file.
        bytes_per_param (int): Number of bytes per parameter for model size calculation.
        batch_size (int): The batch size to simulate for input.
        device (int): GPU index to run the model on. Use ``-1`` for CPU.
        n_trials (int): Number of trials for power and energy measurement.
        extra_attrs (Optional[List[str]]): Additional attributes to write to the file.
        verbose (bool): If True, print detailed information.
    """
    params = model.count_params()
    resource_usage_metrics, inference_time = get_memory_and_time(
        model, batch_size=batch_size, device=device, verbose=verbose
    )
    _, avg_power, avg_energy = get_model_usage_stats(model, device=device, n_trials=n_trials, verbose=verbose)

    ram_metrics = {"system_ram", "gpu_ram"}

    def _resource_component(metric: str, component: str) -> Union[str, int, float]:
        value = resource_usage_metrics.get(metric, "Not measured")
        if isinstance(value, str):
            return value
        component_value = value.get(component)
        if component_value is None:
            return "Not measured"
        return component_value

    def _format_resource(metric: str, component: str, is_ram: bool) -> str:
        value = _resource_component(metric, component)
        if isinstance(value, str):
            return value
        if is_ram:
            return format_bytes(value)
        return f"{value:.2f}%"

    resource_usage_diff = {}
    resource_usage_display: Dict[str, Any] = {}
    for metric, value in resource_usage_metrics.items():
        if isinstance(value, str):
            resource_usage_diff[metric] = value
            if metric in ram_metrics:
                resource_usage_display[metric] = value
            continue

        diff_value = value.get("difference")
        resource_usage_diff[metric] = diff_value if diff_value is not None else "Not measured"

        if metric in ram_metrics:
            display_components: Dict[str, str] = {}
            for component in ("before", "current", "difference"):
                component_value = _resource_component(metric, component)
                if isinstance(component_value, str):
                    display_components[component] = component_value
                else:
                    raw_int = int(round(component_value))
                    display_components[component] = f"{raw_int} B ({format_bytes(component_value)})"
            resource_usage_display[metric] = display_components

    def _format_resource_line(
        metric: str,
        label: str,
        component: str,
        component_label: str,
        is_ram: bool,
    ) -> str:
        if is_ram:
            display_source = resource_usage_display.get(metric)
            if isinstance(display_source, dict):
                text = display_source.get(component, "Not measured")
            elif isinstance(display_source, str):
                text = display_source
            else:
                value = _resource_component(metric, component)
                if isinstance(value, str):
                    text = value
                else:
                    raw_int = int(round(value))
                    text = f"{raw_int} B ({format_bytes(value)})"
            return f"{label} {component_label}: {text}"

        return f"{label} {component_label}: {_format_resource(metric, component, is_ram)}"

    model_stats = {
        "num_params": params,
        "model_size": params * bytes_per_param,
        "flops": get_flops(model),
        "macs": get_macs(model),
        "model_summary": capture_model_summary(model),
        "resource_usage": resource_usage_metrics,
        "resource_usage_diff": resource_usage_diff,
        "resource_usage_display": resource_usage_display,
        "inference_time": inference_time,
        "avg_power": avg_power,
        "avg_energy": avg_energy,
    }

    with open(file_path, "w") as file:
        file.write(f"Number of parameters: {format_number_commas(model_stats['num_params'])}\n")
        file.write(f"Model size: {format_bytes(model_stats['model_size'])}\n")
        file.write(f"FLOPs: {format_number(model_stats['flops'])}FLOPs\n")
        file.write(f"MACs: {format_number(model_stats['macs'])}MACs\n")
        for metric, label, is_ram in (
            ("system_ram", "System RAM", True),
            ("gpu_ram", "GPU RAM", True),
            ("gpu_usage", "GPU usage", False),
            ("cpu_usage", "CPU usage", False),
        ):
            for component, component_label in (
                ("before", "before"),
                ("current", "current"),
                ("difference", "delta"),
            ):
                file.write(f"{_format_resource_line(metric, label, component, component_label, is_ram)}\n")
        file.write(f"Inference time: {format_scientific(model_stats['inference_time'], max_precision=4)} s\n")
        file.write(
            f"Average power consumption: {format_scientific(model_stats['avg_power'], max_precision=4)} W\n"
        )
        file.write(
            f"Average energy consumption: {format_scientific(model_stats['avg_energy'], max_precision=4)} J\n"
        )

        # Write extra attributes
        for attr, value in extra_attrs.items():
            file.write(f"{attr}: {value}\n")

        file.write(f"\nModel summary: {model_stats['model_summary']}\n")
