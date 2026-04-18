"""
Utilities for integrating Optuna hyperparameter optimization with PyTorch models.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gc
import os
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from torchviz import make_dot

from araras.ml.common.device import parse_device_spec
from araras.ml.torch.model import (
    count_params,
    estimate_model_flops,
    estimate_training_memory,
    get_model_stats,
    render_model_stats_report,
    _resolve_input_example,
)
from araras.utils.loading_bar import gen_loading_bar
from araras.utils.misc import format_bytes, format_number
from araras.utils.system import _get_nvidia_smi_data
from araras.utils.verbose_printer import VerbosePrinter
from araras.visualization.configs import config_plt

vp = VerbosePrinter()

_CONSECUTIVE_OOM_ERRORS = 0


def _is_oom_error(exc: BaseException) -> bool:
    """Detect out-of-memory errors across different PyTorch versions.
    
    Checks for both torch.cuda.OutOfMemoryError (newer PyTorch) and RuntimeError
    with "out of memory" in the message (compatibility fallback). This dual check
    ensures we catch OOM across PyTorch versions without assuming API stability.
    
    Args:
        exc: Exception instance to check for OOM characteristics.
    
    Returns:
        True if the exception is an OOM error, False otherwise.
    """
    oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_error and isinstance(exc, oom_error):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _default_prune_on() -> Dict[type, Optional[str]]:
    """Build a default OOM error detection configuration for trial pruning.
    
    Creates a mapping of exception types to optional message substrings that
    identify OOM conditions. This is used in trial optimization to automatically
    prune trials that encounter OOM errors, avoiding re-running identical tests.
    
    Returns:
        Dictionary mapping exception types to message substrings (or None).
        Includes torch.cuda.OutOfMemoryError (if available) and RuntimeError
        with "out of memory" substring as fallback for version compatibility.
    """
    prune_on: Dict[type, Optional[str]] = {}
    oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_error:
        prune_on[oom_error] = None
    prune_on[RuntimeError] = "out of memory"
    return prune_on


def _resize_batch(input_example: Any, batch_size: int) -> Any:
    """Resize batch dimension to target size, allocating fresh zero-filled tensors.
    
    Recursively processes nested data structures (lists, tuples, dicts) to resize
    all tensors' batch dimensions. Creates new zero-filled tensors rather than
    using views, ensuring independence between resized copies. Used in model
    profiling when sampling over multiple batch sizes to create diverse input examples.
    
    Args:
        input_example: Input data that may be a tensor, scalar, or nested
            container of tensors/containers/scalars.
        batch_size: Target batch size for all tensors' first dimension.
    
    Returns:
        Data with all tensors resized to (batch_size, ...), container structure
        preserved. Non-tensor types returned unchanged.
    """
    if torch.is_tensor(input_example):
        shape = input_example.shape
        if not shape:
            return input_example
        # Create fresh zero-filled tensor with target batch size; preserves dtype/device.
        return torch.zeros(
            (batch_size, *shape[1:]),
            dtype=input_example.dtype,
            device=input_example.device,
        )
    if isinstance(input_example, (list, tuple)):
        return type(input_example)(_resize_batch(item, batch_size) for item in input_example)
    if isinstance(input_example, dict):
        return {key: _resize_batch(value, batch_size) for key, value in input_example.items()}
    return input_example


def _infer_batch_size_from_example(example: Any, fallback: int = 1) -> int:
    """Extract batch size from the first tensor found in nested data structures.
    
    Recursively searches through containers (lists, tuples, dicts) to find the
    first tensor and returns its first dimension (batch dimension). This enables
    automatic batch size inference without explicit specification, using a
    heuristic assumption that first dimensions represent batch sizes.
    
    Args:
        example: Input example that may be a tensor, scalar, or nested
            container of tensors/containers/scalars.
        fallback: Default batch size to return if no tensor is found. Defaults to 1.
    
    Returns:
        First dimension of the first tensor encountered, or fallback if no
        tensor found or tensor is 0-dimensional (scalar).
    """
    if torch.is_tensor(example):
        # Return batch size (first dimension) if tensor has dimensions; fallback for scalars.
        return int(example.size(0)) if example.dim() > 0 else fallback
    if isinstance(example, (list, tuple)):
        # Search recursively; return on first tensor found.
        for item in example:
            return _infer_batch_size_from_example(item, fallback)
    if isinstance(example, dict):
        # Search recursively through dict values; return on first tensor found.
        for value in example.values():
            return _infer_batch_size_from_example(value, fallback)
    return fallback


def _estimate_flops_from_example(
    model: torch.nn.Module,
    example: Any,
    *,
    batch_size: Optional[int] = None,
) -> Optional[float]:
    """Estimate FLOPs with graceful fallback, avoiding permanent model modifications.
    
    Temporarily assigns example input as a model attribute for FLOP estimation,
    then restores the original state. This allows estimate_model_flops() to access
    the example while keeping the caller's model unmodified. Returns None on any
    estimation failure to support graceful degradation (e.g., for unsupported
    model architectures).
    
    Batch size is inferred from the example's first tensor if not explicitly provided.
    
    Args:
        model: PyTorch model to profile for FLOPs.
        example: Example input data for FLOP measurement (tensor or nested container).
        batch_size: Optional batch size override. If None, inferred from example.
            Defaults to None.
    
    Returns:
        Estimated FLOPs for the model with the given example, or None if estimation
        fails for any reason (catching all exceptions for robust profiling).
    """
    resolved_batch = batch_size or _infer_batch_size_from_example(example, 1)
    if example is None:
        try:
            return estimate_model_flops(model, batch_size=resolved_batch)
        except Exception:
            return None
    attr_name = "example_input"
    # Preserve any existing attribute to restore later (important for models that use example_input).
    had_attr = hasattr(model, attr_name)
    prev = getattr(model, attr_name, None)
    setattr(model, attr_name, example)
    try:
        return estimate_model_flops(model, batch_size=resolved_batch)
    except Exception:
        return None
    finally:
        # Restore original state to avoid polluting the caller's model.
        if had_attr:
            setattr(model, attr_name, prev)
        else:
            delattr(model, attr_name)


def prune_model_by_config(
    trial: optuna.Trial,
    model: torch.nn.Module,
    thresholds: Dict[str, float],
    *,
    bytes_per_param: int = 8,
    batch_size: int = 1,
    verbose: int = 0,
    input_example: Optional[Any] = None,
) -> None:
    """Prune trial if model exceeds resource constraints, enabling architecture filtering.
    
    Implements early rejection for architectures that violate resource budgets,
    avoiding wasted computation on impractical models. Measures four key statistics
    (parameters, model size, training memory, FLOPs) and compares against thresholds.
    Raises TrialPruned if any metric exceeds its limit, allowing Optuna to
    reallocate computational budget to more promising configurations.
    
    This is especially useful in neural architecture search (NAS) where most sampled
    architectures are infeasible. Early pruning prevents training large or memory-heavy
    models that would fail anyway, saving hours of wasted computation.
    
    Args:
        trial: Optuna Trial object to prune via optuna.TrialPruned exception.
        model: PyTorch model to evaluate against thresholds.
        thresholds: Dict mapping metric names to threshold values. Supported keys:
            - 'param': Maximum parameter count (scalar).
            - 'model_size': Maximum model size in MB.
            - 'memory_mb': Maximum training memory in MB.
            - 'flops': Maximum FLOPs per forward pass.
        bytes_per_param: Bytes used per parameter for size estimation. Defaults to 8
            (assumes float32). Use 4 for float32 parameters, 2 for float16, etc.
        batch_size: Batch size for memory and FLOPs measurement. Defaults to 1.
        verbose: Verbosity level (0=silent, >0=print measured statistics). Defaults to 0.
        input_example: Optional example input for FLOPs/memory estimation. If None,
            will be inferred from the model. Defaults to None.
    
    Raises:
        optuna.TrialPruned: If any measured metric exceeds its threshold.
    
    Notes:
        - If thresholds dict is empty, returns immediately without measurement (NOP).
        - Measurements return None on failure (e.g., unsupported model); these values
          are silently skipped in threshold comparisons (no pruning if unmeasurable).
        - Model is not modified by this function (evaluation only).
    """
    vp.verbose = verbose

    if not thresholds:
        return

    # Measure key model statistics to evaluate against thresholds.
    # Early evaluation helps discard architectures that exceed resource budgets
    # before committing computational resources to training.
    param_count = count_params(model)
    model_size_mb = param_count * bytes_per_param / (1024 * 1024)

    training_memory = estimate_training_memory(
        model,
        input_example=input_example,
        batch_size=batch_size,
        bytes_per_param=bytes_per_param,
    )
    memory_mb = training_memory / (1024 * 1024) if training_memory is not None else None

    if input_example is not None:
        flops_value = _estimate_flops_from_example(model, input_example, batch_size=batch_size)
    else:
        flops_value = _estimate_flops_from_example(model, None, batch_size=batch_size)

    metrics: Dict[str, Optional[float]] = {
        "param": float(param_count),
        "model_size": float(model_size_mb),
        "memory_mb": float(memory_mb) if memory_mb is not None else None,
        "flops": float(flops_value) if flops_value is not None else None,
    }

    if verbose > 0:
        vp.printf(f"Model statistics for trial {trial.number}:", tag="[ARARAS INFO] ", color="blue")
        for key, value in metrics.items():
            if value is None:
                vp.printf(vp.color(f"  {key}: N/A", "blue"))
            else:
                vp.printf(vp.color(f"  {key}: {value:.2f}", "blue"))
        print()

    for key, threshold in thresholds.items():
        value = metrics.get(key)
        if value is None:
            continue
        if value > threshold:
            vp.printf(
                f"Pruning trial {trial.number}: {key} {value:.2f} exceeds {threshold}",
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
            raise optuna.TrialPruned(f"Model exceeded {key} limit")


def plot_model_param_distribution(
    build_model_fn: Callable[[optuna.Trial], Any],
    bytes_per_param: int,
    batch_size: Union[int, Iterable[int]] = 1,
    n_trials: int = 1000,
    fig_save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6),
    csv_path: Optional[str] = None,
    logs_dir: Optional[str] = None,
    corr_csv_path: Optional[str] = None,
    plot_model_dir: Optional[str] = None,
    show_plot: bool = False,
    verbose: int = 1,
    benchmark_training: bool = False,
    device: str = "gpu/0",
    X_train: Optional[Any] = None,
    y_train: Optional[Any] = None,
    input_example: Optional[Any] = None,
) -> None:
    """Sample random models and analyze complexity distributions via histograms and statistics.
    
    Generates a population of random model architectures using a builder function and
    builds comprehensive complexity profiles via parameter counts, memory usage, and FLOPs.
    Produces visualizations and CSV exports for data-driven architecture exploration,
    helping identify realistic design spaces for your hardware and use case.
    
    Primary use cases:
      - Characterizing the relationship between hyperparameters and model complexity.
      - Identifying realistic ranges for parameters (e.g., typical layers, filters).
      - Detecting pathological hyperparameter settings that create extreme models.
      - Benchmarking training speed distributions to estimate total optimization time.
    
    Output artifacts:
      - Histograms of parameter/memory/FLOPs distributions (PNG/PDF).
      - Training latency histograms (if benchmark_training=True).
      - CSV with per-trial metrics (parameters, memory, FLOPs, hyperparameters).
      - Spearman correlation matrix showing which hyperparameters drive complexity.
      - Model architecture visualizations (torchviz graphs) for interesting trials.
      - Per-trial error logs for failures (OOM, unsupported architectures, etc.).
    
    Error handling:
      - OOM errors are tracked separately; study may terminate early if OOM count
        exceeds threshold to avoid extended failure periods on GPU.
      - Other errors are logged per-trial but do not halt the overall study.
      - Per-trial logs are written to logs_dir for post-hoc debugging.
    
    Args:
        build_model_fn: Callable that accepts an Optuna Trial and returns a model.
            Can optionally return a (model, input_example) tuple for custom inputs.
            Called n_trials times with different sampled hyperparameters.
        bytes_per_param: Bytes per parameter for model size estimation. Defaults to 8
            (float32). Use 4 for float32, 2 for float16, etc.
        batch_size: Batch size(s) for profiling. Single int or list of ints. If list,
            profiles memory/FLOPs for each batch size separately to show scaling.
            Defaults to 1.
        n_trials: Number of random model samples to generate and profile.
            Defaults to 1000.
        fig_save_path: Path where histograms are saved. Can be a directory or file
            pattern. If None, figures are not saved. Defaults to None.
        figsize: Matplotlib figure dimensions (width, height). Defaults to (18, 6).
        csv_path: Path to save per-trial metrics CSV. If None, CSV not saved.
            Columns include hyperparameters, parameter count, memory, FLOPs, training time.
        logs_dir: Directory to save per-trial error logs (one file per failed trial).
            Logs include exception message and full traceback. Defaults to None.
        corr_csv_path: Path to save Spearman correlation matrix (CSV). Shows which
            hyperparameters correlate with model complexity. Defaults to None.
        plot_model_dir: Directory to save torchviz model architecture graphs (PNG).
            Graphs are generated for a subset of interesting trials. Defaults to None.
        show_plot: If True, displays plots interactively via plt.show(). Defaults to False.
        verbose: Verbosity level (0=silent, >0=progress bars and messages). Defaults to 1.
        benchmark_training: If True, benchmarks one training epoch on each model to
            estimate training latency. Uses X_train/y_train if provided; otherwise
            synthesizes dummy data from input_example or model inference. Defaults to False.
        device: Device for profiling: 'cpu' or 'gpu/<index>'. Defaults to 'gpu/0'.
        X_train: Optional training features for benchmarking (torch.Tensor). If provided
            with benchmark_training=True, used directly in training loop. Defaults to None.
        y_train: Optional training targets aligned with X_train. If None but X_train
            provided, inferred from model output shape. Defaults to None.
        input_example: Optional example input for FLOP/memory estimation. Defaults to None.
    
    Notes:
        - Study uses Optuna's RandomSampler for unbiased exploration.
        - Batch size normalization removes duplicates if list provided.
        - Model profiling includes handling of various OOM conditions gracefully.
        - Training benchmarks are skipped silently if data unavailable or model incompatible.
    """

    # Benchmarking setup: collect models first, then benchmark smallest/largest
    # This avoids long per-model benchmark times during exploration

    if not show_plot:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    config_plt("double-column")

    # Use Optuna's random sampler to explore the hyperparameter space uniformly.
    # This enables unbiased sampling across all hyperparameter configurations.
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")

    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    # Normalize batch sizes: accept single int or iterable, remove duplicates.
    # This preprocessing ensures consistent handling regardless of input format.
    if isinstance(batch_size, (int, np.integer)):
        batch_sizes = [int(batch_size)]
    elif isinstance(batch_size, (list, tuple)):
        if not batch_size:
            raise ValueError("batch_size iterable must contain at least one value.")
        normalized_batch_sizes = []
        for value in batch_size:
            if not isinstance(value, (int, np.integer)):
                raise TypeError("batch_size iterable must contain only integers.")
            normalized_batch_sizes.append(int(value))
        batch_sizes = list(dict.fromkeys(normalized_batch_sizes))
    else:
        raise TypeError("batch_size must be an int or a list/tuple of ints.")

    if plot_model_dir:
        os.makedirs(plot_model_dir, exist_ok=True)

    def _log_error(trial: optuna.Trial, err: BaseException) -> None:
        """Write trial error details and traceback to per-trial log file.
        
        Creates a log file with the trial's hyperparameters, error message, and
        full traceback for post-hoc debugging. Helps identify which hyperparameter
        combinations lead to failures (e.g., OOM, dimension mismatches).
        """
        if not logs_dir:
            return
        log_file = os.path.join(logs_dir, f"trial_{trial.number}.log")
        with open(log_file, "w") as f:
            f.write(f"Params: {trial.params}\n")
            f.write(f"Error: {err}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())

    def _save_model_plot(model: torch.nn.Module, trial_number: int, example: Any) -> None:
        """Render model computation graph as PNG using torchviz for visual inspection.
        
        Creates a directed graph showing the model's architecture and data flow.
        Useful for debugging NAS output and visualizing hyperparameter-driven
        structural changes.
        """
        if plot_model_dir is None:
            return
        if example is None:
            if verbose > 0:
                vp.printf(
                    "plot_model_dir set but no input_example provided; skipping model plot.",
                    tag="[ARARAS WARNING] ",
                    color="yellow",
                )
            return
        model.eval()
        with torch.no_grad():
            # Invoke model with appropriate unpacking based on input type.
            output = model(example) if not isinstance(example, (tuple, dict)) else (
                model(*example) if isinstance(example, tuple) else model(**example)
            )
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render(os.path.join(plot_model_dir, f"model_{trial_number}"), cleanup=True)

    def _prepare_dummy_from_example(example: Any, batch_size: int, model: torch.nn.Module, device: torch.device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Synthesize dummy training data from example input by inferring output shape.
        
        Creates a tuple of (dummy_x, dummy_y) for training. Dummy_x is resized to the
        target batch size; dummy_y is inferred from model output shape (classification
        vs. regression). Returns (None, None) if example is not suitable (None or non-tensor).
        
        This enables training benchmarks even when explicit training data is unavailable.
        
        Returns:
            Tuple of (dummy_x, dummy_y) ready for training, or (None, None) if unable
            to synthesize from example.
        """
        if example is None:
            return None, None
        if not isinstance(example, torch.Tensor):
            return None, None

        # Resize batch dimension to match target batch size.
        dummy_x = _resize_batch(example, batch_size)
        dummy_x = dummy_x.to(device)

        # Infer output shape from model to determine classification vs. regression.
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_x)

        # Classification task: multi-class logits shape (B, C) where C > 1.
        if isinstance(out, torch.Tensor) and out.ndim >= 2 and out.shape[1] > 1:
            num_classes = out.shape[1]
            dummy_y = torch.zeros((batch_size,), dtype=torch.long, device=device)
            return dummy_x, dummy_y

        # Regression task: use float zeros matching batch size.
        dummy_y = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        return dummy_x, dummy_y

    def _benchmark_training_epoch(
        model: torch.nn.Module,
        batch_size: int,
        device_str: str,
        X_train: Optional[torch.Tensor] = None,
        y_train: Optional[torch.Tensor] = None,
        example: Optional[Any] = None,
        verbose: int = 0,
    ) -> Optional[float]:
        """Benchmark a single training epoch and return elapsed wall-clock time.
        
        Measures the time required to run one full training epoch (forward pass,
        backward pass, optimizer update) on the given model with training data.
        Supports multiple data sources: explicit X_train/y_train, dummy synthesis
        from example input, or automatic inference from model architecture.
        
        Returns elapsed time in seconds, or None if benchmarking fails or is skipped
        (e.g., CUDA unavailable for GPU request, unsupported model).
        
        Args:
            model: PyTorch model to benchmark.
            batch_size: Batch size for DataLoader during training epoch.
            device_str: Device string ('cpu' or 'gpu/<index>').
            X_train: Optional training features tensor. If provided, used directly.
            y_train: Optional training targets. If None but X_train provided, inferred
                from model output. If X_train also None, attempts dummy synthesis.
            example: Example input for dummy data synthesis if X_train unavailable.
            verbose: Verbosity level (0=silent, >0=print decisions).
        
        Returns:
            Elapsed wall-clock time in seconds for one training epoch, or None if
            benchmarking skipped or failed.
        """
        try:
            # Determine device from string spec.
            if device_str == "cpu":
                bench_device = torch.device("cpu")
            elif device_str.startswith("gpu/"):
                gpu_idx = int(device_str.split("/")[1])
                if not torch.cuda.is_available():
                    if verbose > 0:
                        vp.printf(
                            "Skipping benchmark: CUDA is not available for requested GPU device.",
                            tag="[ARARAS WARNING] ",
                            color="yellow",
                        )
                    return None
                bench_device = torch.device(f"cuda:{gpu_idx}")
            else:
                bench_device = torch.device(device_str)

            # Prepare data
            data_x = X_train
            data_y = y_train
            if data_x is not None:
                data_x = data_x.to(bench_device)
                if data_y is None:
                    # Infer target type from model output using a small forward pass
                    with torch.no_grad():
                        sample_out = model(data_x[:1].to(bench_device))
                    if isinstance(sample_out, torch.Tensor) and sample_out.ndim >= 2 and sample_out.shape[1] > 1:
                        data_y = torch.zeros((data_x.shape[0],), dtype=torch.long, device=bench_device)
                    else:
                        data_y = torch.zeros((data_x.shape[0],), dtype=torch.float32, device=bench_device)
                else:
                    data_y = data_y.to(bench_device)
            else:
                # Try to create dummy inputs from example
                data_x, data_y = _prepare_dummy_from_example(example, batch_size, model, bench_device)
                
                # If example also not available, try to automatically infer from model
                if data_x is None or data_y is None:
                    try:
                        inferred_x = _resolve_input_example(model, None, batch_size, device=bench_device)
                        inferred_x = inferred_x.to(bench_device)
                        
                        # Infer target type from model output using a small forward pass
                        with torch.no_grad():
                            sample_out = model(inferred_x[:1])
                        
                        if isinstance(sample_out, torch.Tensor) and sample_out.ndim >= 2 and sample_out.shape[1] > 1:
                            inferred_y = torch.zeros((batch_size,), dtype=torch.long, device=bench_device)
                        else:
                            inferred_y = torch.zeros((batch_size,), dtype=torch.float32, device=bench_device)
                        
                        data_x = inferred_x
                        data_y = inferred_y
                    except Exception:
                        return None

            # Setup training
            criterion = torch.nn.CrossEntropyLoss() if data_y.dtype == torch.long else torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model = model.to(bench_device)
            model.train()

            # Create DataLoader
            dataset = torch.utils.data.TensorDataset(data_x, data_y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Benchmark one epoch
            start = time.time()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if torch.cuda.is_available() and bench_device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start
            return elapsed
        except Exception:
            return None

    trial_numbers: List[int] = []
    param_counts: List[int] = []
    model_sizes_mb: List[float] = []
    flops_counts: List[Optional[float]] = []
    training_memory_map: Dict[int, List[Optional[float]]] = {bs: [] for bs in batch_sizes}
    training_time_map: Dict[int, List[Optional[float]]] = {bs: [] for bs in batch_sizes} if benchmark_training else {}
    collected_params: List[Dict[str, Any]] = []

    progress_iter = range(n_trials)
    if n_trials:
        progress_iter = gen_loading_bar(
            progress_iter,
            description=vp.color("Sampling models", "white"),
            total=n_trials,
            bar_color="white",
        )

    oom_count = 0
    other_error_count = 0

    for _ in progress_iter:
        trial = study.ask()
        model: Optional[torch.nn.Module] = None
        local_input = input_example
        try:
            model_result = build_model_fn(trial)
            if isinstance(model_result, tuple) and len(model_result) == 2:
                model, inferred_input = model_result
                if local_input is None:
                    local_input = inferred_input
            else:
                model = model_result

            if not isinstance(model, torch.nn.Module):
                raise TypeError("build_model_fn must return a torch.nn.Module instance")

            # Profile the model to extract complexity metrics.
            n_params = count_params(model)
            size_mb = (n_params * bytes_per_param) / (1024 * 1024)
            flops_value = _estimate_flops_from_example(model, local_input)

            # Measure training memory for each batch size.
            # This captures how memory scales with batch dimension, important for
            # understanding real-world deployment constraints.
            training_memory_values: Dict[int, Optional[float]] = {}
            training_time_values: Dict[int, Optional[float]] = {}
            for batch in batch_sizes:
                batch_input = local_input
                if batch_input is not None:
                    batch_input = _resize_batch(batch_input, batch)
                memory_bytes = estimate_training_memory(
                    model,
                    input_example=batch_input,
                    batch_size=batch,
                    bytes_per_param=bytes_per_param,
                )
                training_memory_values[batch] = (
                    memory_bytes / (1024 * 1024) if memory_bytes is not None else None
                )
                
                # Benchmark training if enabled
                if benchmark_training:
                    training_time = _benchmark_training_epoch(
                        model=model,
                        batch_size=batch,
                        device_str=device,
                        X_train=X_train,
                        y_train=y_train,
                        example=batch_input,
                        verbose=verbose,
                    )
                    training_time_values[batch] = training_time

            if plot_model_dir:
                try:
                    _save_model_plot(model, trial.number, local_input)
                except Exception as exc:
                    vp.printf(
                        f"Failed to plot model {trial.number}: {exc}",
                        tag="[ARARAS ERROR] ",
                        color="red",
                    )

            trial_numbers.append(trial.number)
            param_counts.append(n_params)
            model_sizes_mb.append(size_mb)
            flops_counts.append(flops_value)
            for batch in batch_sizes:
                training_memory_map[batch].append(training_memory_values.get(batch))
                if benchmark_training:
                    training_time_map[batch].append(training_time_values.get(batch))
            collected_params.append(trial.params)

            study.tell(trial, 0.0)
        except Exception as exc:
            if _is_oom_error(exc):
                oom_count += 1
            else:
                other_error_count += 1
            _log_error(trial, exc)
        finally:
            # Clean up model and cache to avoid memory leaks during long profiling runs.
            # Deleting the model explicitly and clearing GPU cache are critical for
            # preventing cumulative memory growth across many trial iterations.
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Benchmark smallest and largest models if enabled
    if benchmark_training and param_counts:
        vp.printf(
            "Starting training benchmarks on smallest and largest models...",
            tag="[ARARAS INFO] ",
            color="cyan",
        )
        
        # Find indices of smallest and largest models
        min_idx = param_counts.index(min(param_counts))
        max_idx = param_counts.index(max(param_counts))
        
        # Determine device label for display
        if device == "cpu":
            device_label = "CPU"
        elif device.startswith("gpu/"):
            gpu_idx = device.split("/")[1]
            device_label = f"GPU:{gpu_idx}"
        else:
            device_label = device
        
        # Determine data mode
        data_mode = "provided dataset" if X_train is not None else "synthetic/inferred data"
        
        # Print benchmark configuration
        if verbose > 0:
            vp.printf(
                f"Benchmark config: batch_size={batch_sizes[0] if batch_sizes else 1}, device={device_label}, data={data_mode}",
                tag="   [ARARAS INFO] ",
                color="cyan",
            )
        
        # Selected models map
        index_map: Dict[str, int] = {}
        if min_idx == max_idx:
            index_map["smallest/largest"] = min_idx
        else:
            index_map["smallest"] = min_idx
            index_map["largest"] = max_idx
        
        # Rebuild models and benchmark them
        benchmark_times: Dict[str, List[Optional[float]]] = {lbl: [] for lbl in index_map.keys()}
        benchmark_records: List[Dict[str, Any]] = []
        
        for label, idx in index_map.items():
            try:
                if verbose > 0:
                    vp.printf(
                        f"Benchmarking {label} model with params={param_counts[idx]}...",
                        tag="   [ARARAS INFO] ",
                        color="cyan",
                    )
                
                # Create a fixed trial with the collected parameters
                trial_result = optuna.trial.FixedTrial(collected_params[idx])
                
                # Rebuild the model with the same hyperparameters
                model_result = build_model_fn(trial_result)
                if isinstance(model_result, tuple) and len(model_result) == 2:
                    bench_model, _ = model_result
                else:
                    bench_model = model_result
                
                # Benchmark each batch size
                local_input = input_example
                for batch in batch_sizes:
                    batch_input = local_input
                    if batch_input is not None:
                        batch_input = _resize_batch(batch_input, batch)
                    
                    training_time = _benchmark_training_epoch(
                        model=bench_model,
                        batch_size=batch,
                        device_str=device,
                        X_train=X_train,
                        y_train=y_train,
                        example=batch_input,
                        verbose=0,
                    )
                    training_time_map[batch][idx] = training_time
                    benchmark_times[label].append(training_time)
                    benchmark_records.append(
                        {
                            "label": label,
                            "trial_index": idx,
                            "param_count": param_counts[idx],
                            "batch_size": batch,
                            "device": device_label,
                            "training_time_seconds": training_time,
                        }
                    )
                
                del bench_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                if verbose > 0:
                    vp.printf(
                        f"Benchmarking '{label}' model failed: {e}",
                        tag="   [ARARAS WARNING] ",
                        color="yellow",
                    )
        
        # Print benchmark results
        if verbose > 0 and benchmark_times:
            for label, times in benchmark_times.items():
                first_time = next((t for t in times if t is not None), None)
                if first_time is not None:
                    pretty_label = label.replace("/", " ").capitalize()
                    vp.printf(
                        f"{pretty_label} model training time: {first_time:.4f}s",
                        tag="   [ARARAS INFO] ",
                        color="cyan",
                    )
                else:
                    vp.printf(
                        f"{label.replace('/', ' ').capitalize()} model training time: unavailable",
                        tag="   [ARARAS WARNING] ",
                        color="yellow",
                    )

        # Persist benchmark results
        if benchmark_records:
            if csv_path:
                base, ext = os.path.splitext(csv_path)
                benchmark_csv_path = f"{base}_training_benchmark{ext or '.csv'}"
            elif logs_dir:
                os.makedirs(logs_dir, exist_ok=True)
                benchmark_csv_path = os.path.join(logs_dir, "training_benchmark_results.csv")
            else:
                benchmark_csv_path = os.path.abspath("training_benchmark_results.csv")

            pd.DataFrame(benchmark_records).to_csv(benchmark_csv_path, index=False)

            if verbose > 0:
                vp.printf(
                    f"Saved training benchmark results to: {benchmark_csv_path}",
                    tag="[ARARAS INFO] ",
                    color="cyan",
                )

    def _resolve_fig_path(metric: str) -> Optional[str]:
        if not fig_save_path:
            return None
        root, ext = os.path.splitext(fig_save_path)
        if ext:
            return f"{root}_{metric}{ext}"
        return os.path.join(fig_save_path, f"{metric}.png")

    def _plot_hist(values: List[float], title: str, metric: str) -> None:
        if not values:
            return
        plt.figure(figsize=figsize)
        plt.hist(values, bins=50)
        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        out_path = _resolve_fig_path(metric)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()

    _plot_hist(param_counts, "Parameter Counts", "parameters")
    _plot_hist(model_sizes_mb, "Model Sizes (MB)", "model_size_mb")
    _plot_hist([v for v in flops_counts if v is not None], "FLOPs", "flops")
    for batch in batch_sizes:
        values = [v for v in training_memory_map[batch] if v is not None]
        _plot_hist(values, f"Training Memory (MB) - batch {batch}", f"training_memory_mb_bs{batch}")
        if benchmark_training:
            training_times = [v for v in training_time_map[batch] if v is not None]
            _plot_hist(training_times, f"Training Time (s) - batch {batch}", f"training_time_s_bs{batch}")

    if csv_path:
        rows: List[Dict[str, Any]] = []
        for idx, params in enumerate(collected_params):
            row = dict(params)
            row["trial_number"] = trial_numbers[idx]
            row["param_count"] = param_counts[idx]
            row["model_size_mb"] = model_sizes_mb[idx]
            row["flops"] = flops_counts[idx]
            for batch in batch_sizes:
                row[f"training_memory_mb_bs{batch}"] = training_memory_map[batch][idx]
                if benchmark_training:
                    row[f"training_time_s_bs{batch}"] = training_time_map[batch][idx]
            rows.append(row)

        df = pd.DataFrame(rows)
        sort_key = None
        if batch_sizes:
            sort_key = f"training_memory_mb_bs{batch_sizes[0]}"
        if sort_key in df.columns:
            df = df.sort_values(sort_key, ascending=False)
        df.to_csv(csv_path, index=False)
        
        if verbose > 0:
            vp.printf(
                f"Saved search space results to: {csv_path}",
                tag="[ARARAS INFO] ",
                color="cyan",
            )

    if corr_csv_path and collected_params:
        param_df = pd.DataFrame(collected_params)
        param_df["param_count"] = param_counts

        # Encode categorical parameters so they participate in correlation analysis
        categorical_cols = param_df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            param_df = pd.get_dummies(param_df, columns=list(categorical_cols), drop_first=False, dummy_na=False)

        numeric_df = param_df.select_dtypes(include=[np.number])
        if "param_count" in numeric_df.columns:
            corr = (
                numeric_df.corr(method="spearman")["param_count"]
                .drop("param_count")
                .dropna()
                .sort_values(key=abs, ascending=False)
            )
            corr.to_csv(corr_csv_path)

    if oom_count or other_error_count:
        if oom_count:
            vp.printf(
                f"Skipped {oom_count} trial(s) due to out-of-memory errors.",
                tag="[ARARAS WARNING] ",
                color="orange",
            )
        if other_error_count:
            vp.printf(
                f"Skipped {other_error_count} trial(s) due to errors.",
                tag="[ARARAS WARNING] ",
                color="orange",
            )


def set_user_attr_model_stats(
    trial: optuna.Trial,
    model: torch.nn.Module,
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
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Profile a model and persist comprehensive statistics on an Optuna trial.
    
    Measures both structural (parameter count, FLOPs) and runtime (latency, power)
    metrics, then stores them as trial user attributes for later analysis and
    visualization in Optuna studies. Enables detailed post-hoc analysis of the
    relationship between hyperparameters and model performance/efficiency.
    
    Supports dual-device profiling (CPU and GPU) to compare performance across
    deployment targets. Results are formatted as human-readable strings for
    reporting alongside raw numerical values.
    
    Args:
        trial: Optuna Trial object to attach statistics to.
        model: PyTorch model to profile.
        batch_size: Batch size for measurements.
        device: Device to profile on ('cpu', 'gpu/<index>', or 'both/<index>').
        stats_to_measure: Iterable of metric names to compute.
        test_runs: Number of inference runs for resource metrics.
        verbose: Verbosity level (0=quiet, >0=progress bars).
        bytes_per_param: Bytes per parameter for size estimation.
        extra_attrs: Optional dictionary of custom attributes to include.
        input_example: Example input for shape inference.
    
    Returns:
        Dictionary mapping device names ('cpu', 'gpu') to their respective stats
        dictionaries, or None if profiling on that device was not requested.
    
    Raises:
        ValueError: If bytes_per_param < 1 or device spec is invalid.
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

    stats_map: Dict[str, Dict[str, Any]] = {}
    if gpu_stats:
        stats_map["gpu"] = gpu_stats
        trial.set_user_attr("model_stats_gpu", gpu_stats)
    if cpu_stats:
        stats_map["cpu"] = cpu_stats
        trial.set_user_attr("model_stats_cpu", cpu_stats)

    trial.set_user_attr("model_stats", stats_map)

    structural_stats = next((stats for stats in (gpu_stats, cpu_stats) if stats), {})

    def _format_with_unit(
        value: Optional[float], unit: str, formatter: Callable[[float], str]
    ) -> Optional[str]:
        if value is None:
            return None
        human = formatter(value)
        return f"{value} {unit} ({human})"

    def _format_bytes_value(value: Optional[int]) -> Optional[str]:
        if value is None:
            return None
        return f"{value} B ({format_bytes(value)})"

    num_params = structural_stats.get("parameters") if structural_stats else None
    model_size = structural_stats.get("model_size") if structural_stats else None
    flops = structural_stats.get("flops") if structural_stats else None
    summary = structural_stats.get("summary") if structural_stats else None

    num_params_display = _format_with_unit(num_params, "parameters", format_number)
    model_size_display = _format_bytes_value(model_size)
    flops_display = _format_with_unit(flops, "FLOPs", format_number)

    trial.set_user_attr("num_params", num_params)
    trial.set_user_attr("num_params_display", num_params_display or "N/A")
    trial.set_user_attr("model_size", model_size)
    trial.set_user_attr("model_size_display", model_size_display or "N/A")
    trial.set_user_attr("flops", flops)
    trial.set_user_attr("flops_display", flops_display or "N/A")
    trial.set_user_attr("model_summary", summary)

    report_text = render_model_stats_report(
        structural_stats,
        cpu_stats=cpu_stats,
        gpu_stats=gpu_stats,
        extra_attrs=extra_attrs,
    )
    trial.set_user_attr("model_stats_report", report_text)
    if extra_attrs:
        trial.set_user_attr("model_stats_extra_attrs", extra_attrs)

    return {"gpu": gpu_stats, "cpu": cpu_stats}


def log_trial_error(
    trial: optuna.Trial,
    exc: BaseException,
    logs_dir: str,
    prune_on: Optional[Dict[type, Optional[str]]] = None,
    propagate: Optional[Dict[type, Optional[str]]] = None,
    force_crash_oom: int | None = 10,
) -> None:
    """Log and manage trial errors, optionally aborting after repeated OOMs.
    
    Implements sophisticated error handling for hyperparameter optimization,
    distinguishing between prunable errors (e.g., OOM) and fatal errors.
    
    Error handling strategy:
      1. Log full error context (GPU stats, trial params, traceback) for debugging.
      2. Check if error should be propagated (re-raised) as-is. This preserves
         Optuna's exception handling logic for special exceptions like TrialPruned.
      3. Check if error matches pruning criteria. If so, raise TrialPruned to
         signal Optuna to skip the trial without wasting further resources.
      4. Track consecutive OOM errors and abort the entire study if threshold
         exceeded (e.g., after 10 OOMs), signaling systematic issues (e.g.,
         GPU out of memory, model architecture patterns causing OOMs).
    
    Arguments:
        trial: Optuna Trial object.
        exc: The exception to log and handle.
        logs_dir: Directory to write error logs to.
        prune_on: Dictionary mapping exception types to optional message substrings.
            If matched, the trial is pruned (TrialPruned raised).
        propagate: Dictionary mapping exception types to optional message substrings.
            If matched, the exception is re-raised as-is.
        force_crash_oom: Number of consecutive OOMs before aborting the study.
            None to disable. Default 10.
    
    Raises:
        optuna.TrialPruned: If the exception matches the prune_on criteria.
        (Re-raises original exception if matches propagate criteria or if not prunable.)
    """

    prune_on = _default_prune_on() if prune_on is None else prune_on
    propagate = {optuna.exceptions.TrialPruned: None} if propagate is None else propagate

    # Check if the error should be propagated without modification.
    # This preserves Optuna's exception handling for special cases like TrialPruned.
    for exc_type, msg_substr in propagate.items():
        if isinstance(exc, exc_type) and (msg_substr is None or msg_substr in str(exc)):
            raise exc

    # Write comprehensive error log including GPU state and trial context.
    # This information is essential for debugging hyperparameter optimization failures.
    path = os.path.join(logs_dir, f"trial_{trial.number}.log")
    with open(str(path), "w", encoding="utf-8") as log_file:
        try:
            gpu_data = _get_nvidia_smi_data()
            log_file.write("GPU Stats:\n")
            if gpu_data:
                for gpu in gpu_data:
                    log_file.write(
                        f"  GPU {gpu['index']} - {gpu['name']}: "
                        f"used {gpu['used_mb']}MB / {gpu['total_mb']}MB, "
                        f"free {gpu['free_mb']}MB, "
                        f"temp {gpu['temperature']}C, "
                        f"util {gpu['utilization']}%\n"
                    )
            else:
                log_file.write("  No GPU information available\n")
        except Exception as err:  # noqa: BLE001
            log_file.write(f"Failed to collect GPU stats: {err}\n")
        log_file.write(f"Trial: {trial.number}\n")
        log_file.write(f"Params: {trial.params}\n")
        log_file.write(f"User Attributes: {trial.user_attrs}\n")
        log_file.write(f"Exception Type: {type(exc).__name__}\n")
        log_file.write(f"Exception Message: {str(exc)}\n")
        log_file.write("Traceback:\n")
        log_file.write(traceback.format_exc())

    # Track consecutive OOM errors to detect systemic problems.
    # Once threshold is reached, abort the entire study to avoid wasting resources.
    global _CONSECUTIVE_OOM_ERRORS
    if _is_oom_error(exc):
        _CONSECUTIVE_OOM_ERRORS += 1
    else:
        _CONSECUTIVE_OOM_ERRORS = 0

    if force_crash_oom is not None and _CONSECUTIVE_OOM_ERRORS >= force_crash_oom:
        vp.printf(
            f"Reached {_CONSECUTIVE_OOM_ERRORS} consecutive OOM errors. Aborting.",
            tag="[ARARAR ERROR] ",
            color="red",
        )
        os.abort()

    # Check if error should be pruned (trial skipped, allowing optimization to continue).
    for exc_type, msg_substr in prune_on.items():
        message = str(exc).lower()
        if isinstance(exc, exc_type) and (
            msg_substr is None or msg_substr.lower() in message
        ):
            vp.printf(
                (
                    f"Trial {trial.number} failed with {type(exc).__name__}"
                    f"{', message contains ' + repr(msg_substr) if msg_substr else ''}. Pruning."
                ),
                tag="[ARARAS WARNING] ",
                color="yellow",
            )
            raise optuna.TrialPruned() from exc

    raise exc


__all__ = [
    "log_trial_error",
    "plot_model_param_distribution",
    "prune_model_by_config",
    "set_user_attr_model_stats",
]
