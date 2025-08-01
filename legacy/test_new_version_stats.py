from araras.core import *

import time
import tensorflow as tf
import pynvml
import psutil
from inspect import signature
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


def get_flops(model: tf.keras.Model, batch_size: int = 1) -> int:
    """
    Returns the total number of floating point operations (FLOPs).

    The model is traced with dummy inputs matching ``batch_size`` and the
    resulting graph is profiled using TensorFlow's V1 profiler API.

    Args:
        model: Keras model to inspect.
        batch_size: Batch size used for the dummy input. ``1`` by default.

    Returns:
        The FLOP count for a single forward pass.

    Notes:
        TensorFlow's profiler uses graph mode under the hood; therefore the
        model is executed once to build a concrete function before profiling.
    """

    # Supress API deprecation warnings
    logging.getLogger("tensorflow").addFilter(
        lambda r: "tensor_shape_from_node_def_name" not in r.getMessage()
    )

    # 1) Use the *original* structure
    target_structure = model.input  # tensor if single-input, list/tuple/dict otherwise
    flat_tensors = tf.nest.flatten(target_structure)

    # 2) Build specs with same shapes and dtypes
    flat_specs = [tf.TensorSpec([batch_size, *K.int_shape(t)[1:]], dtype=t.dtype) for t in flat_tensors]
    spec_struct = tf.nest.pack_sequence_as(target_structure, flat_specs)

    # 3) Trace with a single positional arg that carries the whole structure
    @tf.function(input_signature=[spec_struct])
    def _forward_fn(x):
        return model(x, training=False)

    concrete = _forward_fn.get_concrete_function()
    opts = ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    info = profile(concrete.graph, options=opts)  # This API was designed for TensorFlow v1
    return info.total_float_ops


def get_macs(model: tf.keras.Model, batch_size: int = 1) -> int:
    """
    Estimates the number of Multiply-Accumulate operations (MACs) required
    for a single forward pass of the model. Assumes 1 MAC = 2 FLOPs.

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
) -> Tuple[int, float]:
    """
    Measures the peak memory usage and average inference time of a Keras model
    on GPU or CPU. The model is first warmed up to account for graph tracing and kernel
    compilation. Subsequent runs are timed while monitoring either the GPU
    memory statistics or the CPU resident set size.

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

    The CPU memory probe occasionally reports zero usage. When this happens, the
    measurement is retried up to two additional times. If all attempts still
    report zero memory, the function returns ``0`` for the peak usage and emits a
    warning.

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        batch_size (int): The batch size to simulate for input. Defaults to 1.
            Measure with batch_size=1 to get base per-sample latency.
        device (int): GPU index to run the model on. Use ``-1`` to run on CPU.
        warmup_runs (int): Number of warm-up runs before timing. Defaults to 10.
        test_runs (int): Number of runs to measure average inference time. Defaults to 50.
        verbose (bool): If True, displays a progress bar during test runs.

    Returns:
        Tuple[int, float]:
            - peak memory usage in bytes (0 if CPU measurement fails after
              several attempts)
            - average inference time in seconds
    """

    def _zeros_like_model_inputs(model: tf.keras.Model, batch_size: int):
        """Create zeros preserving the original input nesting."""
        structure = model.input  # tensor if single input, else nested
        flat = tf.nest.flatten(structure)
        flat_zeros = [tf.zeros([batch_size] + list(K.int_shape(t)[1:]), dtype=t.dtype) for t in flat]
        return tf.nest.pack_sequence_as(structure, flat_zeros)

    dummy_inputs = _zeros_like_model_inputs(model, batch_size)

    # Build a TensorSpec with the same nesting
    def _spec_from_tensor(t):
        shape = [d if d is not None else 1 for d in t.shape]
        return tf.TensorSpec(shape=shape, dtype=t.dtype)

    spec_struct = tf.nest.map_structure(_spec_from_tensor, dummy_inputs)

    @tf.function(input_signature=[spec_struct])
    def infer(x):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"The structure of `inputs`", UserWarning)
            return model(x, training=False)

    use_gpu = device >= 0
    device_str = f"/GPU:{device}" if use_gpu else "/CPU:0"

    if use_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus or device >= len(gpus):
            raise RuntimeError(f"No GPU found for index {device}")

        tf.config.experimental.reset_memory_stats(device_str)

        # Warm-up (first call traces)
        _ = infer(dummy_inputs)
        for _ in range(warmup_runs - 1):
            _ = infer(dummy_inputs)

        times = []
        it = range(test_runs)
        if verbose:
            it = white_track(it, description="Measuring GPU", total=test_runs)

        for _ in it:
            t0 = time.perf_counter()
            out = infer(dummy_inputs)
            _ = tf.nest.flatten(out)[0].numpy()  # force sync
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        peak_mem = tf.config.experimental.get_memory_info(device_str)["peak"]
        return peak_mem, avg_time

    # CPU path
    if not tf.config.list_physical_devices("CPU"):
        raise RuntimeError("No CPU device found")

    def _measure_cpu() -> Tuple[int, float]:
        proc = psutil.Process()
        baseline = proc.memory_info().rss

        with tf.device("/CPU:0"):
            _ = infer(dummy_inputs)
            for _ in range(warmup_runs - 1):
                _ = infer(dummy_inputs)

        peak_rss = baseline
        times = []
        with tf.device("/CPU:0"):
            it = range(test_runs)
            if verbose:
                it = white_track(it, description="Measuring CPU", total=test_runs)
            for _ in it:
                t0 = time.perf_counter()
                out = infer(dummy_inputs)
                _ = tf.nest.flatten(out)[0].numpy()
                times.append(time.perf_counter() - t0)
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss

        return peak_rss - baseline, sum(times) / len(times)

    max_retries = 2
    for attempt in range(max_retries + 1):
        peak_mem, avg_time = _measure_cpu()
        if peak_mem != 0:
            return peak_mem, avg_time
        if attempt < max_retries:
            logger.warning(f"{YELLOW}CPU memory usage measured as 0 bytes, retrying measurement...{RESET}")
        else:
            logger.error(f"{RED}CPU memory usage could not be measured, returning 0.{RESET}")
            return 0, avg_time


def get_model_usage_stats(
    saved_model: str | tf.keras.Model,
    n_trials: int = 10000,
    device: int = 0,
    rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Estimate average power draw and per-inference energy consumption.

    Power measurements come from NVML when ``device`` is a GPU index or from the
    Intel RAPL interface when ``device`` is ``-1``.  The model is executed
    ``n_trials`` times and instantaneous power readings are averaged.

    Warning:
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

    def _make_dummy_for_keras(m: tf.keras.Model):
        # m.input keeps the original structure (tensor, list, tuple, dict)
        spec_structure = m.input
        flat_specs = tf.nest.flatten(spec_structure)
        flat_rand = [
            tf.random.normal([1 if d is None else d for d in t.shape], dtype=t.dtype) for t in flat_specs
        ]
        return tf.nest.pack_sequence_as(spec_structure, flat_rand)

    dummy_inputs = None
    infer: Callable[[], tf.Tensor]

    if is_keras_model:
        dummy_inputs = _make_dummy_for_keras(keras_model)

        def infer():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"The structure of `inputs`", UserWarning)
                return keras_model(dummy_inputs, training=False)

    else:
        # SavedModel branch (already a dict). Keep it consistent:
        _, kwargs = signature.structured_input_signature
        dummy_inputs = tf.nest.map_structure(
            lambda spec: tf.random.normal(
                [1 if d is None else d for d in spec.shape.as_list()], dtype=spec.dtype
            ),
            kwargs,
        )

        def infer():
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
        - Peak memory usage
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
    peak_mem_usage, inference_time = get_memory_and_time(
        model, batch_size=batch_size, device=device, verbose=verbose
    )
    _, avg_power, avg_energy = get_model_usage_stats(model, device=device, n_trials=n_trials, verbose=verbose)

    model_stats = {
        "num_params": params,
        "model_size": params * bytes_per_param,
        "flops": get_flops(model),
        "macs": get_macs(model),
        "model_summary": capture_model_summary(model),
        "peak_memory_usage": peak_mem_usage,
        "inference_time": inference_time,
        "avg_power": avg_power,
        "avg_energy": avg_energy,
    }

    with open(file_path, "w") as file:
        file.write(f"Number of parameters: {format_number_commas(model_stats['num_params'])}\n")
        file.write(f"Model size: {format_bytes(model_stats['model_size'])}\n")
        file.write(f"FLOPs: {format_number(model_stats['flops'])}FLOPs\n")
        file.write(f"MACs: {format_number(model_stats['macs'])}MACs\n")
        file.write(f"Peak memory usage: {format_bytes(model_stats['peak_memory_usage'])}\n")
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
