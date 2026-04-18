"""
Utilities for profiling and analyzing PyTorch models.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gc
import inspect
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.flop_counter import FlopCounterMode

from araras.ml.common.device import parse_device_spec
from araras.utils.loading_bar import gen_loading_bar
from araras.utils.misc import format_bytes, format_number, format_scientific
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


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
    cublas_workspace_config: Optional[str] = None,
) -> Tuple[torch.Generator, Callable[[int], None]]:
    """Seeds common RNGs for best-effort reproducibility and returns DataLoader helpers.

    This function seeds:
      1) Python `random` RNG via `random.seed(seed)`.
      2) NumPy global RNG via `np.random.seed(seed)`.
      3) PyTorch RNG via `torch.manual_seed(seed)` and, if CUDA is available,
         `torch.cuda.manual_seed_all(seed)`.

    It also returns:
      - A `torch.Generator` seeded with `seed`, intended to be passed to
        `torch.utils.data.DataLoader(generator=...)`.
      - A `worker_init_fn(worker_id)` function intended to be passed to
        `torch.utils.data.DataLoader(worker_init_fn=...)` to seed Python and NumPy
        RNGs inside each worker process.

    DataLoader details:
      - When `num_workers > 0`, DataLoader spawns worker processes and reseeds them.
        For reproducible shuffling and worker-side randomness, pass both returned
        values to DataLoader, `generator` and `worker_init_fn`.

    Determinism mode:
      - If `deterministic=True`, this function configures PyTorch to prefer
        deterministic behavior by setting:
          * `torch.use_deterministic_algorithms(True)`
          * `torch.backends.cudnn.benchmark = False`
          * `torch.backends.cudnn.deterministic = True`
      - Deterministic settings can reduce performance and can raise errors if a
        used operation has no deterministic implementation.

    Environment variables:
      - `PYTHONHASHSEED` is not set here. If you need deterministic hashing,
        set it before launching Python, for example:
          `PYTHONHASHSEED=0 python train.py`
        Setting it inside Python is too late to affect hashing done during startup.

      - If `cublas_workspace_config` is not None, this function sets:
          `os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config`
        This can affect determinism for some CUDA linear algebra kernels.
        For maximum reliability, set it in the shell before launching Python and
        before any CUDA context is created.

        Example values:
          - ":0:0"
          - ":4096:2"

    Limitations:
      - Perfect reproducibility is not guaranteed across PyTorch versions, platforms,
        or CPU vs GPU, even with identical seeds.

    Args:
        seed: Seed value used for Python, NumPy, and PyTorch RNGs.
        deterministic: If True, enables deterministic algorithm enforcement and
            cuDNN deterministic settings. Default is False.
        cublas_workspace_config: Value assigned to CUBLAS_WORKSPACE_CONFIG, or None
            to leave it unchanged. Default is None.

    Returns:
        A tuple (g, worker_init_fn):
          - g: A `torch.Generator` seeded with `seed`, pass to DataLoader.
          - worker_init_fn: A function with signature `(worker_id: int) -> None`
            to pass to DataLoader to seed each worker's Python and NumPy RNGs.
    """
    if cublas_workspace_config is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_workspace_config)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return g, worker_init_fn


def clear_torch_session(
    *objs: Any,
    gc_collect: bool = True,
    cuda_empty_cache: bool = True,
    cuda_ipc_collect: bool = True,
    reset_compile: bool = True,
) -> None:
    """Clears common PyTorch state between repeated runs in the same process.

    PyTorch does not have a Keras-style global "session". What usually accumulates
    across runs is (1) Python references to tensors and modules, (2) CUDA allocator
    cached blocks, and (3) compilation caches when using torch.compile.

    This function performs best-effort cleanup:
      - Drops local references to any passed objects (does not delete your caller
        variables). For large objects, you should still `del` them in the caller.
      - Optionally forces Python garbage collection.
      - Optionally releases unoccupied cached GPU memory held by PyTorch's CUDA
        caching allocator.
      - Optionally performs CUDA IPC garbage collection.
      - Optionally resets torch.compile related caches.

    Args:
        *objs: Optional objects to dereference inside this function. This only helps
            if there are no remaining references elsewhere. If you have variables like
            `model`, `optimizer`, `batch`, delete them in your caller for actual frees.
        gc_collect: If True, runs `gc.collect()` to force Python garbage collection.
            Default is True.
        cuda_empty_cache: If True and CUDA is available, calls `torch.cuda.empty_cache()`
            to release unoccupied cached blocks back to the CUDA allocator. Default is True.
        cuda_ipc_collect: If True and CUDA is available, calls `torch.cuda.ipc_collect()`.
            Default is True.
        reset_compile: If True, attempts to clear compilation caches using
            `torch.compiler.reset()` when available, and also `torch._dynamo.reset()`
            when available. Default is True.

    Notes:
        - `torch.cuda.empty_cache()` does not free memory that is still referenced by
          live tensors, and it does not increase the amount of GPU memory available
          to PyTorch. It mainly releases unused cached blocks and can reduce fragmentation.
        - Some GPU memory overhead, such as the CUDA context, cannot be fully released
          without terminating the process.
        - If you need CUDNN workspace or allocator state to fully reset, a process
          restart is the only guaranteed way.

    Returns:
        None.
    """
    _ = objs

    if reset_compile:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
            torch._dynamo.reset()

    if gc_collect:
        gc.collect()

    if torch.cuda.is_available():
        if cuda_empty_cache:
            torch.cuda.empty_cache()
        if cuda_ipc_collect:
            torch.cuda.ipc_collect()

    if gc_collect:
        gc.collect()


def _as_tuple_inputs(input_example: Any) -> Tuple[Any, ...]:
    """Normalize input data into a tuple for unpacking with *args syntax.
    
    Converts various input representations (single tensors, lists, tuples) into
    a consistent tuple format. This enables uniform model invocation using *args,
    avoiding special cases for different input container types and supporting
    models with multiple positional arguments.
    
    Args:
        input_example: Input data that may be a tuple, list, or single value.
            If already a tuple, returned as-is; if a list, converted to tuple;
            otherwise wrapped in a single-element tuple.
    
    Returns:
        A tuple containing the normalized input data, ready for *args unpacking.
    """
    if isinstance(input_example, tuple):
        return input_example
    if isinstance(input_example, list):
        return tuple(input_example)
    return (input_example,)


def _to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors to a device while preserving container structure.
    
    Handles heterogeneous data structures containing tensors, numpy arrays, and
    nested containers. All tensors and arrays are moved/converted; non-tensor
    scalars pass through unchanged. Essential for multi-device inference where
    model parameters and inputs must reside on the same device.
    
    Args:
        data: Input data that may be a tensor, numpy array, scalar, or nested
            container (list, tuple, dict) of the above types.
        device: Target torch device to move tensors/arrays to.
    
    Returns:
        Data with all tensors/arrays on the target device, container structure
        preserved. Scalars and non-tensor types returned unchanged.
    """
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data, device=device)
    if isinstance(data, (list, tuple)):
        return type(data)(_to_device(item, device) for item in data)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    return data


def _iter_tensors(data: Any) -> Iterable[torch.Tensor]:
    """Recursively yield all tensors from nested data structures.
    
    Flattens complex input structures (tuples, lists, dicts) to extract tensor
    components for batch size inference, shape validation, or device checking.
    Non-tensor scalars are skipped silently. Supports arbitrary nesting depths
    and mixed container types.
    
    Args:
        data: Input data that may be a tensor or nested container of tensors
            and other types (lists, tuples, dicts, scalars).
    
    Yields:
        Individual torch.Tensor objects found at any nesting level.
    """
    if torch.is_tensor(data):
        yield data
    elif isinstance(data, dict):
        for value in data.values():
            yield from _iter_tensors(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _iter_tensors(value)


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Infer the device placement of model parameters and buffers.
    
    Models don't explicitly store their device; we infer it by inspecting
    the first parameter or buffer found. This is necessary for correctly
    moving data to the same device before inference. Returns CPU as fallback
    for empty models that have neither parameters nor buffers (rare edge case).
    
    Args:
        model: PyTorch model to inspect for device placement.
    
    Returns:
        The torch.device where the model's tensors reside. Defaults to CPU
        if the model has no parameters or buffers.
    """
    for param in model.parameters():
        return param.device
    for buf in model.buffers():
        return buf.device
    return torch.device("cpu")


def _resolve_dtype(model: torch.nn.Module) -> torch.dtype:
    """Infer the primary data type (float16, float32, etc.) of model tensors.
    
    Examines model parameters and buffers to determine their dtype. Essential
    for creating input examples with matching precision to avoid dtype mismatches
    during inference. Assumes homogeneous dtype (all parameters share one dtype),
    which is standard practice. Defaults to float32 for empty models.
    
    Args:
        model: PyTorch model to inspect for parameter dtype.
    
    Returns:
        The torch.dtype of model parameters. Defaults to torch.float32 if the
        model has no parameters or buffers.
    """
    for param in model.parameters():
        return param.dtype
    for buf in model.buffers():
        return buf.dtype
    return torch.float32


def _resize_batch_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize tensor batch dimension using views to minimize memory allocation.
    
    Key optimization: uses expand() to create a zero-copy view of the first sample
    repeated across the batch dimension, avoiding unnecessary memory allocation
    during profiling. This is critical because profiling overhead should not
    inflate memory estimates. The contiguous() call ensures the result is
    compatible with ops requiring dense layouts (e.g., GEMM operations).
    
    If tensor is already the target batch size, only moves to device. For 0-D
    scalars, moves to device without resizing.
    
    Args:
        tensor: Input tensor with arbitrary batch size (first dimension).
        batch_size: Target batch size for output tensor.
        device: Target device to move tensor to.
    
    Returns:
        Tensor with shape (batch_size, *original_shape[1:]) on target device,
        backed by minimal memory allocation through view operations.
    """
    if tensor.dim() == 0:
        return tensor.to(device=device)
    if tensor.size(0) == batch_size:
        return tensor.to(device=device)
    # Reshape by expanding the first sample to match target batch size.
    # This avoids allocating unnecessary memory for profiling tasks.
    base = tensor[:1].to(device=device)
    return base.expand(batch_size, *base.shape[1:]).contiguous()


def _resize_batch(data: Any, batch_size: int, device: torch.device) -> Any:
    """Recursively resize batch dimension for nested data structures.
    
    Args:
        data: Input data (tensor, container, or scalar).
        batch_size: Target batch size.
        device: Target device.
    
    Returns:
        Data with resized batch dimensions.
    """
    if torch.is_tensor(data):
        return _resize_batch_tensor(data, batch_size, device)
    if isinstance(data, (list, tuple)):
        return type(data)(_resize_batch(item, batch_size, device) for item in data)
    if isinstance(data, dict):
        return {key: _resize_batch(value, batch_size, device) for key, value in data.items()}
    return data


def _prepare_input_example(
    input_example: Any,
    batch_size: int,
    device: torch.device,
) -> Any:
    """Prepare input example by moving to device and resizing batch.
    
    Args:
        input_example: Raw input data provided by user.
        batch_size: Target batch size for the example.
        device: Target device.
    
    Returns:
        Input example ready for model profiling.
    """
    moved = _to_device(input_example, device)
    return _resize_batch(moved, batch_size, device)


def _infer_input_example(
    model: torch.nn.Module,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> Any:
    """Infer or provide example input for a model using a hierarchical strategy.
    
    Enables automatic profiling of models by obtaining or inferring suitable input
    examples. Uses a fallback chain to handle diverse model configurations:
    
    1. Check for user-provided attributes (example_inputs, example_input, example_input_array).
       Models can define these to supply their own example inputs.
    2. Infer shape from the first nn.Linear layer found in the model.
       This is a reasonable heuristic for models with a simple input structure,
       assuming linear layers expose the input feature dimension.
    3. Raise an error if no strategy succeeds.
    
    The hierarchical approach allows maximum flexibility: users can provide hints,
    or the inference works automatically for many common architectures.
    
    Args:
        model: PyTorch model to infer input for.
        batch_size: Batch size to use in the example.
        device: Device to place the example on. Uses model's device if not provided.
    
    Returns:
        Example input tensor with shape (batch_size, ...).
    
    Raises:
        ValueError: If no input can be inferred and no example is provided.
    """
    resolved_device = device or _get_model_device(model)
    dtype = _resolve_dtype(model)
    # First, check if the model has predefined example inputs.
    for attr in ("example_inputs", "example_input", "example_input_array"):
        if hasattr(model, attr):
            example = getattr(model, attr)
            if example is not None:
                return _prepare_input_example(example, batch_size, resolved_device)

    # If no example is provided, attempt to infer input shape from first Linear layer.
    # This is a reasonable heuristic for models with a simple input structure.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return torch.zeros(
                (batch_size, module.in_features), dtype=dtype, device=resolved_device
            )

    raise ValueError(
        "Cannot infer model input. Provide input_example or set model.example_input(s), "
        "or ensure the model contains an nn.Linear to infer (batch_size, in_features)."
    )


def _resolve_input_example(
    model: torch.nn.Module,
    input_example: Optional[Any],
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
) -> Any:
    """Resolve input example by inferring or preparing provided input.
    
    Args:
        model: PyTorch model.
        input_example: User-provided example input, or None to infer.
        batch_size: Batch size for the example.
        device: Target device. Uses model's device if not provided.
    
    Returns:
        Prepared input example ready for inference.
    """
    if input_example is None:
        return _infer_input_example(model, batch_size, device=device)
    resolved_device = device or _get_model_device(model)
    return _prepare_input_example(input_example, batch_size, resolved_device)


def _resolve_torch_device(device: str) -> Tuple[str, Optional[int], torch.device]:
    """Parse device string and validate GPU availability against system.
    
    Converts human-readable device specs (e.g., 'cpu', 'gpu/0') into a tuple
    of (device_kind, gpu_index, torch.device). Validates that requested GPUs
    exist and CUDA is available before returning, failing fast on invalid specs.
    
    Args:
        device: Device specification string such as 'cpu' or 'gpu/0'.
    
    Returns:
        Tuple of (device_kind, gpu_index, torch.device) where device_kind is
        'cpu' or 'gpu', gpu_index is the GPU number (or None for CPU), and
        torch.device is the resolved PyTorch device object.
    
    Raises:
        ValueError: If device spec is malformed or GPU index cannot be resolved.
        RuntimeError: If requested GPU index exceeds available GPUs or CUDA unavailable.
    """
    device_kind, gpu_index = parse_device_spec(device)
    if device_kind == "both":
        raise ValueError("device must be either 'cpu' or 'gpu/<index>'")

    if device_kind == "gpu":
        if gpu_index is None:
            raise ValueError("GPU device index could not be resolved")
        if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
            raise RuntimeError(f"No GPU found for index {gpu_index}")
        torch_device = torch.device(f"cuda:{gpu_index}")
    else:
        torch_device = torch.device("cpu")

    return device_kind, gpu_index, torch_device


def _call_model(model: torch.nn.Module, inputs: Any) -> Any:
    """Invoke model with input unpacking based on container type.
    
    Handles three input styles: dict (unpacked as **kwargs), tuple (unpacked as
    *args), and single value (passed directly). This uniform wrapper enables
    models to accept diverse input formats without caller-side special casing.
    
    Args:
        model: PyTorch model to invoke.
        inputs: Input data: dict for keyword arguments, tuple for positional
            arguments, or single tensor/value for direct passing.
    
    Returns:
        Model output (tensor, tuple, dict, or other container as defined by model).
    """
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def count_params(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model.
    
    Sums the element counts across all parameter tensors. This includes all
    trainable weights and biases but excludes buffers (e.g., batch norm running
    stats) which are not typically counted in parameter statistics.
    
    Args:
        model: PyTorch model to count parameters for.
    
    Returns:
        Total number of scalar parameters across all parameter tensors.
    """
    return int(sum(param.numel() for param in model.parameters()))


def capture_model_summary(
    model: torch.nn.Module,
    *,
    input_example: Optional[Any] = None,
    batch_size: int = 1,
    depth: int = 3,
) -> str:
    """Generate a detailed model architecture summary via torchinfo.
    
    Produces a layer-by-layer breakdown including tensor shapes, parameter
    counts, and memory footprints. Useful for visualizing model structure,
    debugging shape mismatches, and understanding layer-wise complexity.
    
    Args:
        model: PyTorch model to summarize.
        input_example: Example input for shape inference. If None, automatically
            inferred from the model (via Linear layer or model attributes).
        batch_size: Batch size for shape inference. Defaults to 1.
        depth: Maximum layer nesting depth to display. Defaults to 3 (summary
            view); increase for more detailed per-layer breakdowns.
    
    Returns:
        Formatted text summary of model layers and shapes, or error message
        if summary generation fails (e.g., unsupported model architecture).
    """
    input_data = _resolve_input_example(model, input_example, batch_size)
    try:
        result = summary(model, input_data=input_data, depth=depth, verbose=0)
        return str(result)
    except Exception as exc:
        return f"Error generating summary: {exc}"


def run_dummy_inference(
    model: torch.nn.Module,
    input_example: Optional[Any] = None,
    batch_size: int = 1,
    device: str = "cpu",
    warmup_runs: Optional[int] = None,
    runs: int = 1,
    verbose: int = 1,
) -> Tuple[float, float]:
    """Benchmark inference latency with GPU synchronization for accurate timing.
    
    Measures inference latency by running the model on dummy inputs, with critical
    GPU synchronization before/after each measurement to account for asynchronous
    kernel execution. Warmup runs stabilize GPU clocks and CPU caches before
    timing measurements begin, reducing variance and improving accuracy.
    
    GPU synchronization (torch.cuda.synchronize) is essential because GPU kernels
    are queued asynchronously; without it, timing would stop before kernels finish,
    severely underestimating latency. CPU inference does not require synchronization.
    
    Args:
        model: PyTorch model to benchmark.
        input_example: Example input for shape inference. If None, automatically
            inferred from the model.
        batch_size: Batch size for inference runs. Defaults to 1.
        device: Device specification: 'cpu' or 'gpu/<index>'. Defaults to 'cpu'.
        warmup_runs: Number of warm-up runs before measurement (auto-tuned if None).
            Warm-ups stabilize GPU clocks and fill caches before timing begins.
        runs: Number of timed inference runs to measure. Defaults to 1.
        verbose: Verbosity level (0=silent, >0=show progress bar). Defaults to 1.
    
    Returns:
        Tuple of (average_latency_s, peak_latency_s) from measured runs in seconds.
        Average is the mean; peak is the maximum observed latency.
    
    Raises:
        ValueError: If runs < 1 or batch_size < 1.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    device_kind, gpu_index, torch_device = _resolve_torch_device(device)
    device_label = "CPU" if device_kind == "cpu" else f"GPU:{gpu_index}"

    inputs = _resolve_input_example(
        model,
        input_example,
        batch_size,
        device=torch_device,
    )

    # Store original state to avoid corrupting the training script.
    original_device = _get_model_device(model)
    was_training = model.training

    # Move model to target device and set eval mode for consistent inference behavior.
    if original_device != torch_device:
        model.to(torch_device)
    model.eval()

    warmups = max(int(warmup_runs or 0), 0)

    def _execute_once() -> float:
        # Synchronize GPU to ensure kernel launches are completed before timing.
        # This prevents measuring time while kernels are still queued.
        # Without sync, we risk underestimating latency as the timer might stop
        # before GPU work actually completes (asynchronous execution issue).
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        start = time.perf_counter()
        with torch.inference_mode():
            _call_model(model, inputs)
        # Synchronize again to capture all computation before stopping the timer.
        # This ensures we measure wall-clock time until GPU work is truly done.
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        return time.perf_counter() - start

    try:
        for _ in range(warmups):
            _execute_once()

        iterator = range(runs)
        if verbose > 0:
            iterator = gen_loading_bar(
                iterator,
                description=vp.color(f"Calculating latency on {device_label}", "blue"),
                total=runs,
                bar_color="blue",
            )

        measurements = [_execute_once() for _ in iterator]
    finally:
        # Restore original training state and device to avoid corrupting the caller's model.
        # This is critical for maintaining consistency with the user's original setup.
        if was_training:
            model.train()
        if original_device != torch_device:
            model.to(original_device)

    if not measurements:
        return 0.0, 0.0

    average = sum(measurements) / len(measurements)
    peak = max(measurements)
    return average, peak


def estimate_model_flops(
    model: nn.Module,
    batch_size: int = 1,
    type: str = "forward",
) -> float:
    """Estimate floating-point operations (FLOPs) for a model via FlopCounterMode.

    Measures the number of floating-point operations executed during a forward or
    forward+backward pass. Useful for understanding computational complexity and
    comparing model efficiency across architectures. Results depend on batch size,
    supported operator coverage, and code paths taken during execution.

    Forward vs. backward FLOPs:
      - forward: FLOPs for inference (y=f(x)). Lower bound on training cost.
      - backward: FLOPs for forward + gradient computation via backprop. Does not
        include optimizer updates (e.g., Adam step), which add ~2-3x more FLOPs
        depending on optimizer. A complete training step costs approximately:
        forward FLOPs + backward FLOPs + optimizer FLOPs (~2x backward).

    Input handling:
      - First checks for model attributes: example_inputs, example_input, example_input_array.
      - If not found, infers a 2D input (batch_size, in_features) from the first
        nn.Linear layer in the model.
      - Falls back to ValueError if no input can be resolved.

    Limitations:
      - Custom ops and Triton kernels may be undercounted if their FLOP formulas
        are not registered with FlopCounterMode.
      - Branches and conditional logic execute normally; counted FLOPs reflect
        the taken code path for the given input.

    Args:
        model: PyTorch model to measure (nn.Module).
        batch_size: Batch size for the input example. Defaults to 1.
        type: Computation mode: "forward" for inference FLOPs, or "backward"
            for forward+backward FLOPs. Defaults to "forward".

    Returns:
        Estimated number of floating-point operations for one pass with the
        specified batch_size.

    Raises:
        ValueError: If batch_size < 1, type not in {"forward", "backward"},
            or inputs cannot be inferred.
        TypeError: If model output contains no tensors (required for backward mode).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    mode = type.strip().lower()
    if mode not in {"forward", "backward"}:
        raise ValueError('type must be "forward" or "backward"')

    def _call_model(target: nn.Module, inputs: Any) -> Any:
        # Support multiple invocation styles: positional args, kwargs, or single value.
        if isinstance(inputs, tuple):
            return target(*inputs)
        if isinstance(inputs, dict):
            return target(**inputs)
        return target(inputs)

    def _extract_tensor(out: Any) -> torch.Tensor:
        # Recursively find first tensor in output, supporting various return types.
        # Models may return tensors, lists, tuples, or dicts of tensors.
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)) and out:
            return _extract_tensor(out[0])
        if isinstance(out, dict) and out:
            return _extract_tensor(next(iter(out.values())))
        raise TypeError("Model output is not a Tensor or a container holding a Tensor.")

    inputs = _infer_input_example(model, batch_size)
    was_training = model.training

    # Forward-only FLOPs measurement: set eval mode and compute inference FLOPs.
    if mode == "forward":
        model.eval()
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                _ = _call_model(model, inputs)
        if was_training:
            model.train()
        return float(fc.get_total_flops())

    # Full training step FLOPs: forward + backward. Set training mode to capture
    # all operations including dropout, batch norm, and gradient computation.
    model.train()
    model.zero_grad(set_to_none=True)
    fc = FlopCounterMode(display=False)
    with fc:
        out = _call_model(model, inputs)
        out_tensor = _extract_tensor(out)
        loss = out_tensor.float().mean()
        loss.backward()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return float(fc.get_total_flops())


def get_model_flops_table_forward(
    model: nn.Module,
    *,
    input_example: Optional[Any] = None,
    batch_size: int = 1,
) -> str:
    """Generate a per-layer FLOPs breakdown table for a forward pass.
    
    Produces a formatted table showing FLOPs contributed by each layer (to depth 5).
    Useful for identifying bottleneck operations and understanding layer-wise
    computational complexity. More detailed than estimate_model_flops().
    
    Args:
        model: PyTorch model to measure (nn.Module).
        input_example: Optional example input. If None, automatically inferred
            from the model. Defaults to None.
        batch_size: Batch size for the probe input. Defaults to 1.
    
    Returns:
        Formatted table string from FlopCounterMode.get_table() showing per-layer
        FLOPs breakdown with columns for operation names, counts, and totals.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    inputs = _resolve_input_example(model, input_example, batch_size)
    was_training = model.training
    model.eval()
    try:
        # Use no_grad instead of inference_mode to avoid known zero-FLOP cases.
        with torch.no_grad():
            fc = FlopCounterMode(depth=5, display=False)
            with fc:
                _call_model(model, inputs)
        return fc.get_table()
    finally:
        if was_training:
            model.train()


def get_model_flops_table_backward(
    model: nn.Module,
    *,
    input_example: Optional[Any] = None,
    batch_size: int = 1,
) -> str:
    """Return a FLOPs table for forward + backward using FlopCounterMode.

    Args:
        model: The PyTorch model to measure.
        input_example: Optional example input. If not provided, inferred.
        batch_size: Batch size for the probe input.

    Returns:
        A tabulated string (from FlopCounterMode.get_table()) describing FLOPs.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    def _extract_tensor(out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)) and out:
            return _extract_tensor(out[0])
        if isinstance(out, dict) and out:
            return _extract_tensor(next(iter(out.values())))
        raise TypeError("Model output is not a Tensor or a container holding a Tensor.")

    inputs = _resolve_input_example(model, input_example, batch_size)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        fc = FlopCounterMode(depth=5, display=False)
        with fc:
            out = _call_model(model, inputs)
            out_tensor = _extract_tensor(out)
            loss = out_tensor.float().mean()
            loss.backward()
        return fc.get_table()
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()


def estimate_training_memory(
    model: torch.nn.Module,
    input_example: Optional[Any] = None,
    *,
    batch_size: int = 1,
    bytes_per_param: int = 4,
    optimizer_state_dtype: torch.dtype = torch.float32,
) -> Optional[int]:
    """Estimate training memory usage in bytes based on a forward + backward pass.
    
    Comprehensive memory model includes:
      - Model parameters (frozen during training)
      - Gradients for backpropagation
      - Optimizer state (momentum, variance, etc.)
      - Intermediate activations saved for backward pass
    
    Executes a single forward and backward pass with hooks to track intermediate
    tensors. The memory estimate assumes typical optimizer use (e.g., Adam),
    where each trainable parameter needs additional state buffers.
    
    Args:
        model: PyTorch model to profile.
        input_example: Example input for shape inference. If not provided,
            inferred from the model.
        batch_size: Batch size for the forward/backward pass.
        bytes_per_param: Bytes per parameter (4 for float32, 2 for float16, etc.).
        optimizer_state_dtype: Data type assumed for optimizer state buffers
            (e.g., momentum, variance in Adam). Affects per-param memory estimate.
    
    Returns:
        Total memory in bytes, or None if estimation fails.
    """
    if bytes_per_param < 1:
        raise ValueError("bytes_per_param must be at least 1")

    inputs = _resolve_input_example(model, input_example, batch_size)

    # Estimate memory for model parameters and gradients.
    param_bytes = count_params(model) * bytes_per_param
    grad_bytes = sum(
        param.numel() * bytes_per_param
        for param in model.parameters()
        if param.requires_grad
    )
    # Estimate optimizer state (e.g., momentum, variance buffers for Adam).
    # Assumes one additional buffer per trainable parameter of the given dtype size.
    # This is a simplification: Adam actually uses 2 buffers per param (momentum + variance),
    # but this provides a reasonable ballpark estimate for memory planning.
    optimizer_state_bytes = sum(
        param.numel() * torch.empty((), dtype=optimizer_state_dtype).element_size()
        for param in model.parameters()
        if param.requires_grad
    )
    saved_tensor_bytes = 0

    # Hooks to track intermediate tensors saved for the backward pass.
    # This accumulates memory used by activations and intermediate values.
    # These are critical for memory planning since they grow with batch size
    # and can dominate total memory in deep networks.
    def _pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved_tensor_bytes
        if torch.is_tensor(tensor):
            saved_tensor_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def _unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _extract_tensor(out: Any) -> torch.Tensor:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)) and out:
            return _extract_tensor(out[0])
        if isinstance(out, dict) and out:
            return _extract_tensor(next(iter(out.values())))
        raise TypeError("Model output is not a Tensor or a container holding a Tensor.")

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        # Execute forward and backward pass with hooks to measure intermediate tensor memory.
        # The forward pass creates activations; backward pass reuses some and creates others.
        with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
            outputs = _call_model(model, inputs)
            out_tensor = _extract_tensor(outputs)
            if out_tensor.requires_grad:
                loss = out_tensor.float().mean()
                loss.backward()
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()

    total_bytes = param_bytes + grad_bytes + optimizer_state_bytes + saved_tensor_bytes
    return int(total_bytes)


def get_model_stats(
    model: torch.nn.Module,
    batch_size: int = 1,
    device: str = "gpu/0",
    *,
    stats_to_measure: Iterable[str] = (
        "parameters",
        "model_size",
        "flops",
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
    input_example: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect comprehensive structural and runtime statistics for a PyTorch model.
    
    Measures model complexity (parameter count, FLOPs), memory footprint, and
    actual runtime performance including power consumption. Caches intermediate
    results to avoid redundant computation when measuring multiple metrics.
    
    Structural metrics are computed once per model:
      - parameters: Total parameter count
      - model_size: Estimated memory for model weights
      - flops: Floating-point operations for forward/backward pass
      - summary: Detailed layer-by-layer architecture summary
    
    Runtime metrics are measured through repeated inference runs:
      - inference_latency: Actual inference time
      - resource metrics: CPU/GPU utilization, memory, power consumption
    
    GPU-only metrics are skipped on CPU devices and return None.
    
    Args:
        model: PyTorch model to profile.
        batch_size: Batch size for measurements.
        device: Device to run on ('cpu' or 'gpu/<index>').
        stats_to_measure: Iterable of metric names to compute.
        test_runs: Number of inference runs for resource metrics.
        verbose: Verbosity level (0=quiet, >0=progress bars).
        bytes_per_param: Bytes per parameter for size estimation.
        input_example: Example input for shape inference.
    
    Returns:
        Dictionary mapping metric names to their measured values.
    
    Raises:
        ValueError: If test_runs < 1, bytes_per_param < 1, or unknown metric requested.
    """

    if test_runs < 1:
        raise ValueError("test_runs must be at least 1")

    if bytes_per_param < 1:
        raise ValueError("bytes_per_param must be at least 1")

    device_kind, gpu_index, _torch_device = _resolve_torch_device(device)

    resolved_gpu_index = gpu_index if gpu_index is not None else 0
    device_label = "CPU" if device_kind == "cpu" else f"GPU:{gpu_index}"

    stats_sequence = list(dict.fromkeys(stats_to_measure))
    results: Dict[str, Any] = {
        "batch_size": batch_size,
        "parameters_batch_size": batch_size,
    }

    # Cache computed values to avoid redundant work when multiple stats use the same input.
    # This optimization is important for expensive operations like input inference or FLOPs calculation.
    parameter_cache: Optional[int] = None
    resolved_input_example: Optional[Any] = None

    def _get_parameter_count() -> int:
        nonlocal parameter_cache
        if parameter_cache is None:
            parameter_cache = count_params(model)
        return parameter_cache

    def _get_input_example() -> Any:
        nonlocal resolved_input_example
        if resolved_input_example is None:
            resolved_input_example = _resolve_input_example(model, input_example, batch_size)
        return resolved_input_example

    def _estimate_flops_with_inputs() -> Optional[float]:
        example = _get_input_example()
        attr_name = "example_input"
        had_attr = hasattr(model, attr_name)
        prev = getattr(model, attr_name, None)
        setattr(model, attr_name, example)
        try:
            return estimate_model_flops(model, batch_size=batch_size)
        except Exception:
            return None
        finally:
            if had_attr:
                setattr(model, attr_name, prev)
            else:
                delattr(model, attr_name)

    def _measure_resource_metric(metric_name: str) -> Any:
        if metric_name in GPU_ONLY_METRICS and device_kind != "gpu":
            return None

        aggregation = RESOURCE_METRIC_AGGREGATIONS.get(metric_name)
        if aggregation is None:
            return None

        example = _get_input_example()

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
                input_example=example,
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

    for stat in stats_sequence:
        if stat == "parameters":
            results[stat] = _get_parameter_count()
        elif stat == "model_size":
            param_count = _get_parameter_count()
            results[stat] = int(param_count * bytes_per_param)
        elif stat == "flops":
            results[stat] = _estimate_flops_with_inputs()
        elif stat == "summary":
            results[stat] = capture_model_summary(
                model,
                input_example=_get_input_example(),
                batch_size=batch_size,
            )
        elif stat == "inference_latency":
            average, peak = run_dummy_inference(
                model,
                input_example=_get_input_example(),
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
                "Unknown statistic "
                f"'{stat}' requested\nAccepted values: "
                f"{list(RESOURCE_METRIC_AGGREGATIONS.keys()) + ['parameters', 'model_size', 'flops', 'summary', 'inference_latency']}"
            )

    return results


def render_model_stats_report(
    structural_stats: Dict[str, Any],
    *,
    cpu_stats: Optional[Dict[str, Any]] = None,
    gpu_stats: Optional[Dict[str, Any]] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a human-readable textual report summarizing collected model statistics.
    
    Formats collected structural and runtime metrics into a comprehensive report
    suitable for logging or display. Handles missing data gracefully, showing "N/A"
    for unavailable metrics. Supports dual device profiling (CPU and GPU) with
    separate sections for each.
    
    The report includes model complexity, device-specific performance metrics,
    and energy consumption estimates (power * latency).
    
    Args:
        structural_stats: Dictionary with model structure metrics (parameters, size, FLOPs, etc.).
        cpu_stats: Optional dictionary with CPU runtime metrics.
        gpu_stats: Optional dictionary with GPU runtime metrics.
        extra_attrs: Optional dictionary of additional attributes to include in report.
    
    Returns:
        Formatted report as a single string suitable for printing or file output.
    """

    extra_attrs = extra_attrs or {}
    structural_stats = structural_stats or {}

    def _format_plain(value: Optional[Any]) -> str:
        return "N/A" if value is None else str(value)

    def _format_engineering(value: Optional[Any]) -> str:
        return "N/A" if value is None else format_number(value, precision=2)

    def _format_bytes_value(value: Optional[Any]) -> str:
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
    lines.append(
        f"Model Size: {_format_plain(model_size_bytes)} B ({_format_bytes_value(model_size_bytes)})"
    )

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
            f"    - System Memory: {_format_plain(gpu_system_memory)} B ({_format_bytes_value(gpu_system_memory)})"
        )
        lines.append(f"    - GPU Memory: {_format_plain(gpu_memory)} B ({_format_bytes_value(gpu_memory)})")
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
            f"    - System Memory: {_format_plain(cpu_system_memory)} B ({_format_bytes_value(cpu_system_memory)})"
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
    model: torch.nn.Module,
    file_path: str,
    *,
    batch_size: int = 1,
    device: str = "both/0",
    stats_to_measure: Iterable[str] = (
        "parameters",
        "model_size",
        "flops",
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
    input_example: Optional[Any] = None,
) -> None:
    """Profile a PyTorch model comprehensively and persist human-readable statistics to file.
    
    Enables end-to-end model profiling with flexible device targeting and metric
    selection. Captures both structural metrics (parameter count, memory footprint,
    FLOPs) and runtime performance (latency, resource utilization, power consumption).
    
    The profiling workflow:
      1. Resolve device configuration (CPU, GPU, or both) and validate availability.
      2. Invoke get_model_stats() to collect all requested metrics on specified device(s).
      3. Aggregate results into a human-readable report using render_model_stats_report().
      4. Write formatted report to file with optional metadata.
    
    Dual-device profiling:
      When device="both/<index>", profiles on both CPU and GPU (specified GPU) to
      facilitate direct performance comparison. CPU results and GPU results are
      presented side-by-side in the output for easy comparison. This is useful for
      deployment decisions and optimization target selection.
    
    Output format:
      The generated report is a structured text file containing:
        - Model complexity metrics (parameter count, model size, FLOPs)
        - Per-device performance sections (latency, memory, power, utilization)
        - Energy consumption estimates (power × latency)
        - Layer-by-layer architecture summary
        - Custom attributes (if provided)
      
      Metrics that fail to compute (e.g., GPU unavailable, custom ops) display "N/A".
    
    Args:
        model: PyTorch model to profile. Will be moved to specified device for
            measurement and then restored to original state.
        file_path: Absolute or relative path to write the report to. Parent
            directories are created if necessary.
        batch_size: Batch size for all profiling measurements. Larger values
            provide more realistic memory/latency estimates but increase profiling time.
        device: Device specification for profiling:
            - "cpu": Profile on CPU only.
            - "gpu/<index>": Profile on GPU only (e.g., "gpu/0").
            - "both/<index>": Profile on both CPU and GPU (e.g., "both/0").
        stats_to_measure: Iterable of metric names to compute. Supported metrics:
            Structural: "parameters", "model_size", "flops", "summary"
            Runtime: "inference_latency", "cpu_util_percent", "cpu_power_rapl_w",
                     "ram_used_bytes", "ram_util_percent", "gpu_util_percent",
                     "gpu_mem_used_bytes", "gpu_power_w"
            GPU-only metrics are skipped on CPU and return N/A.
        test_runs: Number of inference runs to perform for runtime metrics.
            Higher values reduce variance but increase profiling duration.
        verbose: Verbosity level:
            0 = Silent, no progress output.
            >0 = Display progress bars for long-running measurements.
        bytes_per_param: Estimated bytes per parameter for model size calculation.
            Common values: 4 (float32), 2 (float16), 8 (float64).
        extra_attrs: Optional dictionary of custom attributes to append to report.
            Useful for storing metadata like model name, training date, etc.
            Example: {"model_name": "ResNet50", "framework": "PyTorch"}
        input_example: Optional example input for shape inference. If not provided,
            automatically inferred from the model (via example_input attribute or
            first nn.Linear layer).
    
    Returns:
        None. Results are written directly to file_path.
    
    Raises:
        ValueError: If bytes_per_param < 1 or device spec is invalid (e.g.,
            "gpu/<index>" with no GPU available).
        IOError: If the file cannot be written (permission denied, full disk, etc.).
        RuntimeError: If the requested GPU device does not exist.
    
    Notes:
        - Profiling can be slow for large models or high test_runs. Consider
          reducing test_runs for quick profiling cycles.
        - GPU profiling requires CUDA-capable hardware and torch.cuda.is_available().
        - The function restores the model to its original device and training state
          after profiling to avoid corrupting the caller's workflow.
        - Power measurements (RAPL) require supported CPU/GPU and may return N/A
          on unsupported hardware.
        - Model summaries use torchinfo and may fail for complex or custom models.
    
    Examples:
        Profile a model on GPU and save results:
            >>> model = MyModel()
            >>> write_model_stats_to_file(
            ...     model,
            ...     "model_profile.txt",
            ...     device="gpu/0",
            ...     batch_size=32,
            ...     test_runs=20
            ... )
        
        Profile on both CPU and GPU with custom metadata:
            >>> write_model_stats_to_file(
            ...     model,
            ...     "comparison.txt",
            ...     device="both/0",
            ...     batch_size=16,
            ...     extra_attrs={"model": "ResNet50", "date": "2025-01-22"}
            ... )
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
        "input_example": input_example,
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


def save_model_as_torchscript(
    model: torch.nn.Module,
    path: str,
    *,
    device: str = "cpu",
) -> None:
    """Save a PyTorch model as a TorchScript file.

    Converts the model to TorchScript, which compiles it to an optimized
    intermediate representation. The resulting .pt file can be loaded and
    executed in C++ or without Python dependencies, enabling deployment
    in resource-constrained environments.

    The exporter moves the model to the requested device and switches it to
    eval mode. Restoring the original device and training mode afterward avoids
    leaving a training script in a corrupted state (e.g., the model stuck on
    CPU or in eval mode). Be careful if you remove this restore logic.
    
    Args:
        model: PyTorch model to export.
        path: File path to save the TorchScript (.pt file).
        device: Device to compile on ('cpu' or 'gpu/<index>').
    """
    _device_kind, _gpu_index, torch_device = _resolve_torch_device(device)

    orig_device = _get_model_device(model)
    orig_training = model.training
    try:
        model.eval()
        model_device = model.to(torch_device)
        ts = torch.jit.script(model_device)
        torch.jit.save(ts, path)
    finally:
        model.to(orig_device)
        model.train(orig_training)


def save_model_as_exported_program(
    model: torch.nn.Module,
    path: Union[str, Path],
    export_args: Optional[Union[torch.Tensor, Tuple[Any, ...]]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    dynamic_batch: bool = False,
    strict: bool = False,
    for_inference: bool = True,
) -> None:
    """Export a model using torch.export.export and save as a .pt2 file.

    The torch.export API produces a portable, hardware-independent IR that can
    be optimized and deployed across different backends. Dynamic shapes allow
    the same exported program to handle variable batch sizes at runtime.

    Input handling:
      - If export_args is omitted, an input example is inferred from the model.
      - When dynamic_batch=True, the example batch size must be >= 2 to avoid
        the 0/1 specialization rule in torch.export.
    
    State restoration:
      - Export moves the model to the export device and switches it to eval
        mode. Restoring the original device and training mode afterward avoids
        leaving a training script in a corrupted state (e.g., model stuck on CPU
        or in eval mode). Be careful if you remove this restore logic.
    
    Decomposition:
      - When for_inference=True, run_decompositions() converts high-level ops
        to lower-level implementations for better portability and optimization.
    
    Args:
        model: PyTorch model to export.
        path: File path to save the exported program (.pt2 file).
        export_args: Positional arguments for the model. If None, inferred.
        export_kwargs: Keyword arguments for the model. If None, empty.
        device: Device to export on ('cpu' or 'gpu/<index>').
        dynamic_batch: If True, batch dimension is marked as dynamic.
        strict: If True, enforce strict mode (no graph breaks).
        for_inference: If True, run decompositions for better portability.
    """

    def _resolve_export_device() -> torch.device:
        if isinstance(device, torch.device):
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise RuntimeError(f"No GPU found for index {device.index}")
            return device
        _kind, _gpu_index, torch_device = _resolve_torch_device(str(device))
        return torch_device

    def _forward_param_names(target: torch.nn.Module) -> List[str]:
        try:
            sig = inspect.signature(target.forward)
        except (TypeError, ValueError):
            return []
        names = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                return []
            names.append(param.name)
        return names

    def _split_inputs(example: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if isinstance(example, dict):
            return (), dict(example)
        if isinstance(example, tuple):
            return example, {}
        if isinstance(example, list):
            return tuple(example), {}
        return (example,), {}

    def _build_dynamic_shapes(
        target: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Union[Tuple[Optional[Dict[int, Any]], ...], Dict[str, Dict[int, Any]]]:
        batch_dim = torch.export.Dim("batch")

        def _shape_for(value: Any) -> Optional[Dict[int, Any]]:
            if torch.is_tensor(value) and value.dim() > 0:
                return {0: batch_dim}
            return None

        if not kwargs:
            dyn_args = tuple(_shape_for(arg) for arg in args)
            if all(shape is None for shape in dyn_args):
                raise ValueError("dynamic_batch=True but no tensor inputs were found.")
            return dyn_args

        names = _forward_param_names(target)
        if not names:
            raise ValueError("dynamic_batch=True but forward signature could not be inspected.")
        if len(args) > len(names):
            raise ValueError("dynamic_batch=True but export_args length exceeds forward signature.")

        dyn_kwargs: Dict[str, Dict[int, Any]] = {}
        for idx, arg in enumerate(args):
            shape = _shape_for(arg)
            if shape is not None:
                dyn_kwargs[names[idx]] = shape
        for key, value in kwargs.items():
            if key not in names:
                raise ValueError(
                    f"dynamic_batch=True but kwarg '{key}' is not in forward signature."
                )
            shape = _shape_for(value)
            if shape is not None:
                dyn_kwargs[key] = shape

        if not dyn_kwargs:
            raise ValueError("dynamic_batch=True but no tensor inputs were found.")
        return dyn_kwargs

    def _enforce_dynamic_batch_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        for tensor in _iter_tensors(args):
            if tensor.dim() > 0 and tensor.size(0) <= 1:
                raise ValueError(
                    "dynamic_batch=True requires example inputs with batch >= 2 "
                    f"to avoid 0/1 specialization; got batch={tensor.size(0)} "
                    f"for tensor shape {tuple(tensor.shape)}."
                )
        for tensor in _iter_tensors(kwargs):
            if tensor.dim() > 0 and tensor.size(0) <= 1:
                raise ValueError(
                    "dynamic_batch=True requires example inputs with batch >= 2 "
                    f"to avoid 0/1 specialization; got batch={tensor.size(0)} "
                    f"for tensor shape {tuple(tensor.shape)}."
                )

    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    torch_device = _resolve_export_device()

    if export_kwargs is None:
        export_kwargs = {}
    if export_args is None:
        if export_kwargs:
            export_args = ()
        else:
            inferred_batch = 2 if dynamic_batch else 1
            inferred = _resolve_input_example(
                model, None, batch_size=inferred_batch, device=torch_device
            )
            export_args, export_kwargs = _split_inputs(inferred)
    elif isinstance(export_args, dict) and not export_kwargs:
        export_args, export_kwargs = (), dict(export_args)

    export_args = _as_tuple_inputs(export_args)
    export_args = tuple(_to_device(arg, torch_device) for arg in export_args)
    export_kwargs = {k: _to_device(v, torch_device) for k, v in export_kwargs.items()}
    if dynamic_batch:
        _enforce_dynamic_batch_inputs(export_args, export_kwargs)

    orig_device = _get_model_device(model)
    orig_training = model.training
    try:
        model.eval()
        model.to(torch_device)
        if dynamic_batch:
            dynamic_shapes = _build_dynamic_shapes(model, export_args, export_kwargs)
            ep = torch.export.export(
                model,
                args=export_args,
                kwargs=export_kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
        else:
            ep = torch.export.export(
                model,
                args=export_args,
                kwargs=export_kwargs,
                strict=strict,
            )

        if for_inference and hasattr(ep, "run_decompositions"):
            ep = ep.run_decompositions(decomp_table={})

        torch.export.save(ep, str(export_path))
    finally:
        model.to(orig_device)
        model.train(orig_training)


__all__ = [
    "capture_model_summary",
    "clear_torch_session",
    "count_params",
    "estimate_model_flops",
    "estimate_training_memory",
    "get_model_flops_table_backward",
    "get_model_flops_table_forward",
    "get_model_stats",
    "render_model_stats_report",
    "run_dummy_inference",
    "write_model_stats_to_file",
    "save_model_as_torchscript",
    "save_model_as_exported_program",
]
