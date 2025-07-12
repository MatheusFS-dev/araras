"""
This module provides utilities to calculate the number of floating-point operations (FLOPs)

Functions:
    - get_flops: Calculates the total number of floating-point operations (FLOPs) needed
    - get_macs: Estimates the number of Multiply-Accumulate operations (MACs) required
    - get_memory_and_time: Measures the peak memory usage and average inference time of a Keras model

Example:
    >>> from araras.keras.utils.profiler import get_flops
    >>> get_flops(...)
"""
from araras.commons import *

import time
import tensorflow as tf
import psutil
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


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
    opts["output"] = "none" # Supress report
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
            progress_iter = tqdm(
                progress_iter,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
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
                progress_iter = tqdm(
                    progress_iter,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
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
            logger.warning(
                f"{YELLOW}CPU memory usage measured as 0 bytes, retrying measurement...{RESET}"
            )
        else:
            logger.error(
                f"{RED}CPU memory usage could not be measured, returning 0.{RESET}"
            )

    return peak_mem, avg_time
