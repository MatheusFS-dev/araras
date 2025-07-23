"""Helpers for measuring Keras model resource usage.

This module gathers statistics such as FLOPs, MACs, peak memory usage, and
average inference latency. The utilities are intended for use with small tests
and benchmarks and do not alter model behaviour.
"""

from __future__ import annotations

import logging
import time
from inspect import signature
from typing import Callable, List, Optional, Tuple
import warnings

import psutil
import pynvml
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from araras.core import *
from araras.ml.model.utils import capture_model_summary
from araras.utils.misc import (
    format_bytes,
    format_number,
    format_number_commas,
    format_scientific,
)

def get_flops(model: tf.keras.Model, batch_size: int = 1) -> int:
    """Return the total number of floating point operations (FLOPs).

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

    # Suppress noisy shape warnings emitted by TensorFlow
    logging.getLogger("tensorflow").addFilter(
        lambda r: "tensor_shape_from_node_def_name" not in r.getMessage()
    )

    # 1) Start from the original input structure
    target_structure = model.input  # tensor if single-input, list/tuple/dict otherwise
    flat_tensors = tf.nest.flatten(target_structure)

    # 2) Build TensorSpecs preserving shapes and dtypes
    flat_specs = [tf.TensorSpec([batch_size, *K.int_shape(t)[1:]], dtype=t.dtype) for t in flat_tensors]
    spec_struct = tf.nest.pack_sequence_as(target_structure, flat_specs)

    # 3) Trace with a single positional arg containing the full structure
    @tf.function(input_signature=[spec_struct])
    def _forward_fn(x):
        return model(x, training=False)

    concrete = _forward_fn.get_concrete_function()
    opts = ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    info = profile(concrete.graph, options=opts)  # This API was designed for TensorFlow v1
    return info.total_float_ops


def get_macs(model: tf.keras.Model, batch_size: int = 1) -> int:
    """Return an estimate of the required multiply-accumulate operations.

    The estimation assumes that a single MAC is equivalent to two FLOPs.

    Args:
        model: Keras model to inspect.
        batch_size: Batch size used for the dummy input.

    Returns:
        The estimated number of MACs for one forward pass.
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
    """Measure peak memory usage and average inference latency.

    The model is first warmed up to account for graph tracing and kernel
    compilation. Subsequent runs are timed while monitoring either the GPU
    memory statistics or the CPU resident set size.

    Args:
        model: Keras model to benchmark.
        batch_size: Batch size used for the dummy input.
        device: GPU index or ``-1`` for CPU execution.
        warmup_runs: Number of warm-up iterations run before measurement.
        test_runs: Number of timed runs used to compute the average latency.
        verbose: If ``True``, display a progress bar during measurement.

    Returns:
        Tuple[int, float]:
            Peak memory usage in bytes and the average inference time in
            seconds.

    Notes:
        Warm-up runs prevent one-time initialisation from skewing the results.
        When running on CPU the memory probe can occasionally report ``0``. In
        such cases the measurement is retried twice before giving up.

    Warning:
        CPU memory reporting may fail and return ``0`` even after retries.
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

        # Warm-up so that graph tracing and kernel compilation are excluded
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

    # ----- CPU execution path -----
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
    """Estimate average power draw and per-inference energy consumption.

    Power measurements come from NVML when ``device`` is a GPU index or from the
    Intel RAPL interface when ``device`` is ``-1``.  The model is executed
    ``n_trials`` times and instantaneous power readings are averaged.

    Args:
        saved_model: Path to a SavedModel directory, a ``.keras`` file, or an
            already loaded ``tf.keras.Model``.
        n_trials: Number of inference trials to run.
        device: GPU index to query via NVML or ``-1`` for CPU measurements.
        rapl_path: Path to the RAPL energy counter used for CPU power readings.
        verbose: If ``True``, display a progress bar during the trials.

    Returns:
        Tuple[float, float, float]:
            Average runtime per inference, average power in watts and average
            energy consumed per inference in joules.

    Raises:
        RuntimeError: If NVML initialisation fails when measuring GPU power.
        ValueError: If ``device`` is neither ``-1`` nor a valid GPU index.

    Notes:
        CPU power measurement requires Linux, Intel CPUs and access to the RAPL
        counters (often root privileges). When dummy inputs do not match the
        model signature, Keras may emit a ``UserWarning`` which is suppressed.
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

    # NVML is initialised inside the measurement loop only when GPU power is queried

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
        # Preserve the nested input structure when creating random tensors
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
        # SavedModel branch (already a dict). Create random tensors matching the signature
        _, kwargs = signature.structured_input_signature
        dummy_inputs = tf.nest.map_structure(
            lambda spec: tf.random.normal(
                [1 if d is None else d for d in spec.shape.as_list()], dtype=spec.dtype
            ),
            kwargs,
        )

        def infer():
            return signature(**dummy_inputs)


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
    """Persist model statistics to ``file_path``.

    Statistics include parameter count, FLOPs, MACs, memory usage, inference
    time and power metrics. Additional ``extra_attrs`` may be written after the
    default statistics.

    Args:
        model: Model to analyse.
        file_path: Destination path for the text file.
        bits_per_param: Bit depth assumed per parameter when estimating size.
        batch_size: Batch size used when measuring performance metrics.
        device: GPU index used for measurement, or ``-1`` for CPU.
        n_trials: Number of inference runs used for the power measurement.
        extra_attrs: Optional mapping of attribute names to values written after
            the default statistics.
        verbose: If ``True``, print detailed information during measurement.

    Notes:
        Extra attributes can be used to record custom metrics such as accuracy
        alongside the default statistics.
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
