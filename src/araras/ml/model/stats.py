from araras.core import *

import time
import tensorflow as tf
import psutil

from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

import pynvml

from araras.ml.model.utils import capture_model_summary
from araras.utils.misc import format_number, format_bytes, format_scientific, format_number_commas


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
) -> Tuple[int, float]:
    """
    Measures the peak memory usage and average inference time of a Keras model
    on GPU or CPU.

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
    warning in red.

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
    # Prepare dummy inputs matching model.inputs
    shapes = [(batch_size,) + tuple(K.int_shape(inp)[1:]) for inp in model.inputs]
    dummy_inputs = [tf.zeros(shape, dtype=inp.dtype) for shape, inp in zip(shapes, model.inputs)]

    @tf.function
    def infer(*args):
        return model(list(args), training=False)

    use_gpu = device >= 0
    device_str = f"/GPU:{device}" if use_gpu else "/CPU:0"

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
            t0 = time.perf_counter()
            out = infer(*dummy_inputs)
            _ = out.numpy()
            times.append(time.perf_counter() - t0)
        avg_time = sum(times) / len(times)

        peak_mem = tf.config.experimental.get_memory_info(device_str)["peak"]
        return peak_mem, avg_time

    # CPU path
    if not tf.config.list_physical_devices("CPU"):
        raise RuntimeError("No CPU device found")

    def _measure_cpu() -> Tuple[int, float]:
        """Run the CPU measurement loop and return (peak_mem, avg_time)."""
        proc = psutil.Process()
        baseline = proc.memory_info().rss

        # warmup on CPU
        cpu_index = 0
        with tf.device(f"/CPU:{cpu_index}"):
            _ = infer(*dummy_inputs)
            for _ in range(warmup_runs - 1):
                _ = infer(*dummy_inputs)

        peak_rss = baseline
        times = []
        with tf.device(f"/CPU:{cpu_index}"):
            progress_iter = range(test_runs)
            if verbose:
                progress_iter = white_track(
                    progress_iter,
                    description="Measuring CPU",
                    total=test_runs,
                )
            for _ in progress_iter:
                t0 = time.perf_counter()
                out = infer(*dummy_inputs)
                _ = out.numpy()
                times.append(time.perf_counter() - t0)
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss

        avg_time = sum(times) / len(times)
        peak_mem = peak_rss - baseline
        return peak_mem, avg_time

    max_retries = 2
    peak_mem = 0
    avg_time = 0.0
    for attempt in range(max_retries + 1):
        peak_mem, avg_time = _measure_cpu()
        if peak_mem != 0:
            break
        if attempt < max_retries:
            logger.warning(f"{YELLOW}CPU memory usage measured as 0 bytes, retrying measurement...{RESET}")
        else:
            logger.error(f"{RED}CPU memory usage could not be measured, returning 0.{RESET}")

    return peak_mem, avg_time


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
    bits_per_param: int,
    batch_size: int,
    device: int = 0,
    n_trials: int = 1000,
    extra_attrs: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """
    Write model statistics to a file.

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        file_path (str): The path to the output file.
        bits_per_param (int): Number of bits per parameter for model size calculation.
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
        "model_size": params * bits_per_param,
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
