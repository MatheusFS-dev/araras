from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union

import os
import time

import psutil
import subprocess
from datetime import datetime
from threading import Thread

import tensorflow as tf

from araras.utils.misc import format_bytes, format_number_commas
from araras.utils.verbose_printer import VerbosePrinter
from araras.utils.misc import clear

vp = VerbosePrinter()

MetricSnapshot = Dict[str, Optional[float]]
MetricExtractors = Dict[str, Callable[[List[Dict[str, Any]]], Optional[float]]]
MetricSamples = Dict[str, Dict[str, List[float]]]
MetricStatistics = Dict[str, Union[List[Union[int, float]], Union[int, float], None]]
MetricSummary = Dict[str, Union[str, Dict[str, MetricStatistics]]]
TCallableReturn = TypeVar("TCallableReturn")

# ———————————————————————————————————————————————————————————————————————————— #
#                                     Utils                                    #
# ———————————————————————————————————————————————————————————————————————————— #


def format_metric_summary_line(
    label: str,
    before: Union[None, str, int, float],
    during: Union[None, str, int, float],
    delta: Union[None, str, int, float],
    *,
    is_byte_metric: bool = False,
    percent_precision: int = 2,
) -> str:
    """Format metric statistics into a single descriptive line.

    Args:
        label (str): Human readable label for the metric (e.g. ``"System RAM"``).
        before (Union[None, str, int, float]): Representative "before" statistic, typically the
            maximum observed prior to executing the workload.
        during (Union[None, str, int, float]): Representative "during" statistic, typically the
            maximum observed while the workload was executing.
        delta (Union[None, str, int, float]): Difference between the "during" and "before"
            statistics (e.g. ``during - before``).
        is_byte_metric (bool): Flag indicating whether the values represent bytes.
        percent_precision (int): Decimal precision when formatting percentage metrics.

    Returns:
        str: A formatted single-line summary such as::

            System RAM: 14,828,216,975 B (13.81 GB) - 14,834,859,049 B (13.82 GB) = 6,642,074 B (6.33 MB)

        Metrics lacking numeric data return ``"<label>: Not measured"``.

    Raises:
        ValueError: If ``percent_precision`` is negative.
    """

    if percent_precision < 0:
        raise ValueError("percent_precision must be non-negative")

    def _is_not_measured(value: Union[None, str, int, float]) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() == "not measured":
            return True
        return False

    def _format_component(value: Union[None, str, int, float]) -> str:
        if isinstance(value, str):
            text = value.strip()
            return text or "Not measured"
        if value is None:
            return "Not measured"
        if is_byte_metric:
            raw_int = int(round(float(value)))
            raw_text = f"{format_number_commas(raw_int)} B"
            human_text = format_bytes(value)
            if human_text.lower().startswith("invalid input"):
                return raw_text
            return f"{raw_text} ({human_text})"
        try:
            return f"{float(value):.{percent_precision}f}%"
        except (TypeError, ValueError):
            return "Not measured"

    before_missing = _is_not_measured(before)
    during_missing = _is_not_measured(during)
    delta_missing = _is_not_measured(delta)

    if before_missing and during_missing and delta_missing:
        return f"{label}: Not measured"

    fragments: List[str] = []

    def _append_fragment(text: str, operator: Optional[str] = None) -> None:
        if not text:
            return
        if operator and fragments:
            fragments.append(f"{operator} {text}")
        else:
            fragments.append(text)

    if not during_missing:
        _append_fragment(_format_component(during))

    if not before_missing:
        _append_fragment(_format_component(before), "-")

    if not delta_missing:
        _append_fragment(_format_component(delta), "=")

    if not fragments:
        return f"{label}: Not measured"

    return f"{label}: {' '.join(fragments)}"


def _get_nvidia_smi_data():
    """
    Retrieves GPU information using nvidia-smi command.

    Returns:
        list: List of GPU information dictionaries or empty list if failed
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            gpu_data = []
            lines = result.stdout.strip().split("\n")

            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 7:
                    index, name, total_mb, used_mb, free_mb, temp, util = parts[:7]

                    gpu_info = {
                        "index": int(index),
                        "name": name,
                        "total_mb": float(total_mb),
                        "used_mb": float(used_mb),
                        "free_mb": float(free_mb),
                        "temperature": temp if temp != "[Not Supported]" else "N/A",
                        "utilization": util if util != "[Not Supported]" else "N/A",
                    }
                    gpu_data.append(gpu_info)

            return gpu_data
    except Exception:
        return []


def _print_tensorflow_info():
    """Print TensorFlow configuration information."""
    
    print(f"TensorFlow Configuration")
    print("=" * 80)
    print(f"Version        : {tf.__version__}")

    if tf.test.is_built_with_cuda():
        print("CUDA Support: " + vp.color("Yes", "green"))
        build_info = tf.sysconfig.get_build_info()
        print(f"CUDA Version   : {build_info.get('cuda_version', 'Unknown')}")
        print(f"cuDNN Version  : {build_info.get('cudnn_version', 'Unknown')}")
    else:
        print("CUDA Support: " + vp.color("No", "red"))

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.test.gpu_device_name()


def _print_gpu_table(gpu_data):
    """Print GPU information in nvidia-smi style table format."""
    if not gpu_data:
        print(vp.color("No GPU data available", "red"))
        return

    print(f"\nGPU Information")
    print("=" * 80)

    # Header
    header = f"{'GPU':<3} {'Name':<25} {'Memory Usage':<20} {'Temp':<6} {'Util':<6}"
    print(f"{header}")
    print("-" * 80)

    # GPU rows
    for gpu in gpu_data:
        # Memory calculations
        total_gb = gpu["total_mb"] / 1024
        used_gb = gpu["used_mb"] / 1024

        # Format memory usage with color coding
        memory_str = f"{used_gb:6.1f}GB / {total_gb:6.1f}GB"

        # Format temperature
        temp_str = f"{gpu['temperature']}C" if gpu["temperature"] != "N/A" else "N/A"

        # Format utilization
        util_str = f"{gpu['utilization']}%" if gpu["utilization"] != "N/A" else "N/A"

        # Print row
        row = (
            f"{gpu['index']:<3} "
            f"{gpu['name'][:24]:<25} "
            f"{memory_str:<20} "
            f"{temp_str:<6} "
            f"{util_str:<6}"
        )
        print(row)


def get_gpu_info() -> None:
    """Print detailed TensorFlow and GPU configuration information.

    The output mirrors ``nvidia-smi`` style tables and includes TensorFlow build
    metadata, individual GPU statistics, memory consumption, and, when
    available, temperature plus utilisation metrics.

    Notes:
        Run this helper in an environment where ``nvidia-smi`` is available for
        the richest output. Missing utilities degrade the report gracefully.

    Examples:
        >>> get_gpu_info()
    """
    # Header with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(vp.color(f"TensorFlow GPU Monitor - {timestamp}", "blue"))
    print(vp.color(f"{'=' * 80}", "blue"))

    # TensorFlow configuration
    _print_tensorflow_info()

    # Get GPU data from nvidia-smi
    gpu_data = _get_nvidia_smi_data()

    # Print GPU table
    _print_gpu_table(gpu_data)

    # Print memory summary
    # _print_memory_summary(gpu_data)

    print(vp.color(f"\n{'=' * 80}", "blue"))


def setup_gpu_env(
    *,
    visible_device_indices: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
    memory_limit_mb: Optional[Union[int, float, Dict[Union[str, int], Union[int, float]]]] = None,
    memory_growth: Optional[bool] = None,
    op_determinism: Optional[bool] = None,
    xla_jit: Optional[bool] = None,
    intra_op_threads: Optional[int] = None,
    inter_op_threads: Optional[int] = None,
    env_variables: Optional[Dict[str, Union[str, int, float]]] = None,
    show_cuda_summary: bool = False,
    verbosity: int = 1,
    clear_screen: bool = True,
) -> Dict[str, Any]:
    """C
    onfigure TensorFlow GPU runtime options and related environment flags.

    The helper wraps common CUDA and TensorFlow configuration knobs—device
    visibility, allocator behaviour, determinism, and thread pools—and prints
    colourised status updates through :class:`~araras.utils.verbose_printer.VerbosePrinter`.
    Call it before creating TensorFlow objects so that device selection and
    virtual-device caps take effect.
    

    Notes:
        * All environment variables set here are process scoped. Restart Python
          to undo changes.
        * ``memory_limit_mb`` overrides ``memory_growth`` on devices where both
          are requested; the function warns when this happens.
        * Use ``env_variables`` to set any environment variables you need
          (for example, ``CUDA_VISIBLE_DEVICES``, ``TF_FORCE_GPU_ALLOW_GROWTH``,
          ``TF_GPU_ALLOCATOR`` and ``XLA_FLAGS``) before TensorFlow initialises.

    Warning:
        Using the tf API sometimes does not guarantee the desired effect, that is,
        it may fail silently. So, prefer the usage of environment variables.

    Examples:
        Enable memory growth with verbose logs::

            >>> setup_gpu_env(memory_growth=True, verbosity=1)

        Pin TensorFlow to GPU ``0`` with a 4 GiB cap and deterministic kernels::

            >>> setup_gpu_env(
            ...     visible_device_indices=[0],
            ...     memory_limit_mb={0: 4096},
            ...     env_variables={
            ...         "CUDA_VISIBLE_DEVICES": "0",
            ...         "TF_DETERMINISTIC_OPS": "1",
            ...         "TF_CUDNN_DETERMINISM": "1",
            ...     },
            ...     verbosity=2,
            ... )

    Args:
        visible_device_indices (Optional[Union[str, int, Iterable[Union[str, int]]]]):
            Logical device IDs passed to :func:`tf.config.set_visible_devices`
            so TensorFlow initialises only those GPUs.
        memory_limit_mb (Optional[Union[int, float, Dict[Union[str, int], Union[int, float]]]]):
            Virtual-device caps in MiB set with
            :func:`tf.config.set_logical_device_configuration`. Pass a scalar to
            apply the same limit to all visible GPUs or a dictionary for
            per-device limits.
        memory_growth (Optional[bool]): Enables per-device demand-driven VRAM
            allocation via :func:`tf.config.experimental.set_memory_growth` when
            ``True`` (default); ``False`` reverts to eager allocation.
        op_determinism (Optional[bool]): When ``True``, calls
            :func:`tf.config.experimental.enable_op_determinism` to enforce
            determinism (availability depends on TensorFlow version).
        xla_jit (Optional[bool]): When set, calls
            :func:`tf.config.optimizer.set_jit` to enable/disable XLA JIT
            compilation globally.
        intra_op_threads (Optional[int]): Calls
            :func:`tf.config.threading.set_intra_op_parallelism_threads` to cap
            intra-op CPU threads.
        inter_op_threads (Optional[int]): Calls
            :func:`tf.config.threading.set_inter_op_parallelism_threads` to cap
            inter-op scheduling threads.
        env_variables (Optional[Dict[str, Union[str, int, float]]]): Additional
            environment variables applied verbatim (for example
            ``{"CUDA_VISIBLE_DEVICES": "0", "TF_GPU_THREAD_COUNT": 4}``).
        show_cuda_summary (bool): When ``True`` invokes :func:`get_gpu_info` to
            print a post-configuration GPU summary.
        verbosity (int): Verbosity level passed to :class:`VerbosePrinter`.
            ``0`` suppresses info logs; higher values emit more detail.
        clear_screen (bool): When ``True`` clears the console before printing.

    Returns:
        Dict[str, Any]: Summary containing environment changes, TensorFlow
        configuration outcomes, warnings, and errors.

    Raises:
        TypeError: If argument types do not match the expected signatures.
        ValueError: If numeric arguments are invalid (for example negative
            limits or verbosity).
    """

    if not isinstance(verbosity, int):
        raise TypeError("verbosity must be an int")
    if verbosity < 0:
        raise ValueError("verbosity must be >= 0")

    if visible_device_indices is not None and isinstance(visible_device_indices, (list, tuple, set)):
        visible_device_indices = list(visible_device_indices)

    if memory_limit_mb is not None and isinstance(memory_limit_mb, dict):
        normalized_limits: Dict[int, float] = {}
        for key, value in memory_limit_mb.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise TypeError("memory_limit_mb mapping keys must be integers") from exc
            try:
                limit = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("memory_limit_mb mapping values must be numeric") from exc
            if limit <= 0:
                raise ValueError("memory_limit_mb values must be positive")
            normalized_limits[idx] = limit
        memory_limit_mapping: Optional[Dict[int, float]] = normalized_limits
    else:
        memory_limit_mapping = None
        if memory_limit_mb is not None:
            try:
                numeric_limit = float(memory_limit_mb)
            except (TypeError, ValueError) as exc:
                raise TypeError("memory_limit_mb must be numeric") from exc
            if numeric_limit <= 0:
                raise ValueError("memory_limit_mb must be positive")
            memory_limit_mb = numeric_limit

    if intra_op_threads is not None and intra_op_threads <= 0:
        raise ValueError("intra_op_threads must be positive")

    if inter_op_threads is not None and inter_op_threads <= 0:
        raise ValueError("inter_op_threads must be positive")

    tag = "[ARARAS] "
    previous_verbosity = vp.verbose
    vp.verbose = verbosity

    summary: Dict[str, Any] = {
        "environment": {},
        "tensorflow": {},
        "warnings": [],
        "errors": [],
    }

    def _log_info(message: str) -> None:
        vp.printf(message, tag=tag, color="cyan")

    def _log_success(message: str) -> None:
        message = f"Successfully {message[0].lower() + message[1:]}"
        vp.printf(message, tag=tag, color="green")

    def _log_warning(message: str) -> None:
        summary["warnings"].append(message)
        prev = vp.verbose
        try:
            if prev == 0:
                vp.verbose = 1
            vp.printf(message, tag=tag, color="yellow")
        finally:
            vp.verbose = prev

    def _log_error(message: str) -> None:
        summary["errors"].append(message)
        prev = vp.verbose
        try:
            if prev == 0:
                vp.verbose = 1
            vp.printf(message, tag=tag, color="red")
        finally:
            vp.verbose = prev

    def _normalize_device_spec(
        spec: Optional[Union[str, int, Iterable[Union[str, int]]]],
        *,
        allow_empty: bool = False,
    ) -> Optional[List[int]]:
        if spec is None:
            return None
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, str):
            cleaned = spec.strip()
            if not cleaned:
                return [] if allow_empty else None
            tokens = [tok.strip() for tok in cleaned.split(",") if tok.strip()]
            indices: List[int] = []
            for tok in tokens:
                try:
                    indices.append(int(tok))
                except ValueError as exc:
                    raise TypeError("Device identifiers must be integers") from exc
            return indices
        try:
            indices = [int(value) for value in spec]
        except Exception as exc:  # noqa: BLE001 - broad to convert type errors into TypeError
            raise TypeError("Device identifiers must be integers") from exc
        return indices

    # Helpers for GPU discovery and pretty names
    def _tf_device_friendly_name(device: Any) -> str:
        """Best-effort friendly device name for a TensorFlow PhysicalDevice."""
        try:
            details = tf.config.experimental.get_device_details(device)
        except Exception:
            details = {}
        name = None
        if isinstance(details, dict):
            name = details.get("device_name") or details.get("device_desc")
        if not name:
            name = getattr(device, "name", None)
        return str(name) if name else "GPU"

    try:
        if clear_screen:
            clear()
        _log_info("Starting TensorFlow GPU environment configuration")

        # --- Environment variables (generic) ---------------------------------------
        def _set_env(var: str, value: Optional[str]) -> None:
            if value is None:
                return
            os.environ[var] = value
            summary["environment"][var] = value
            # Include GPU names when setting CUDA_VISIBLE_DEVICES via env_variables
            if var == "CUDA_VISIBLE_DEVICES":
                try:
                    tokens = [t.strip() for t in str(value).split(",") if t.strip()]
                    indices = []
                    for t in tokens:
                        try:
                            indices.append(int(t))
                        except Exception:
                            # Non-integer tokens are ignored for naming
                            pass
                    name_map = {d.get("index"): d.get("name") for d in _get_nvidia_smi_data()}
                    names = [name_map.get(i, f"GPU {i}") for i in indices]
                    pretty = ", ".join(names) if names else None
                except Exception:
                    pretty = None
                if pretty:
                    _log_success(f"Set {var}={value} ({pretty})")
                else:
                    _log_success(f"Set {var}={value}")
            else:
                _log_success(f"Set {var}={value}")
        if env_variables:
            for key, value in env_variables.items():
                _set_env(str(key), str(value))

        # No implicit environment mutation beyond env_variables

        # --- TensorFlow runtime configuration ------------------------------------
        if visible_device_indices is not None:
            parsed_indices = _normalize_device_spec(visible_device_indices, allow_empty=True)
        else:
            parsed_indices = None

        physical_gpus = tf.config.list_physical_devices("GPU")
        summary["tensorflow"]["physical_device_count"] = len(physical_gpus)
        # Avoid extra detection printouts per user preference

        if parsed_indices is not None:
            if not physical_gpus:
                _log_warning("No physical GPUs detected; cannot restrict visible devices")
            else:
                try:
                    selected_devices = []
                    for idx in parsed_indices:
                        if idx >= len(physical_gpus) or idx < 0:
                            raise IndexError(idx)
                        selected_devices.append(physical_gpus[idx])
                    tf.config.set_visible_devices(selected_devices, "GPU")
                    summary["tensorflow"]["visible_devices"] = parsed_indices
                    names = [
                        _tf_device_friendly_name(physical_gpus[idx]) for idx in parsed_indices
                        if 0 <= idx < len(physical_gpus)
                    ]
                    pretty_names = ", ".join(names)
                    _log_success(
                        f"Restricted TensorFlow to GPU indices: {parsed_indices} ({pretty_names})"
                    )
                    physical_gpus = selected_devices
                except IndexError as exc:
                    raise ValueError(
                        "visible_device_indices contains an out-of-range GPU index",
                    ) from exc
                except RuntimeError as exc:
                    _log_error(f"Failed to set visible devices: {exc}")

        configured_limits: Dict[str, float] = {}
        if memory_limit_mapping is not None or isinstance(memory_limit_mb, (int, float)):
            if not physical_gpus:
                _log_warning("No physical GPUs detected; skipping memory limit configuration")
            else:
                if memory_limit_mapping is not None:
                    out_of_range = [
                        k for k in memory_limit_mapping.keys() if k < 0 or k >= len(physical_gpus)
                    ]
                    if out_of_range:
                        _log_warning(
                            "memory_limit_mb contains indices out of range for currently visible GPUs: "
                            f"{sorted(out_of_range)}"
                        )
                for idx, device in enumerate(physical_gpus):
                    if memory_limit_mapping is not None:
                        limit_value = memory_limit_mapping.get(idx)
                        if limit_value is None:
                            continue
                    else:
                        limit_value = float(memory_limit_mb)  # type: ignore[arg-type]
                    try:
                        tf.config.experimental.set_virtual_device_configuration(
                            device,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit_value)],
                        )
                        configured_limits[device.name] = limit_value
                        dev_name = _tf_device_friendly_name(device)
                        _log_success(
                            f"Capped GPU {dev_name} memory to {limit_value:.0f} MiB",
                        )
                    except RuntimeError as exc:
                        dev_name = _tf_device_friendly_name(device)
                        _log_error(f"Failed to cap memory on {dev_name}: {exc}")

        if configured_limits:
            summary["tensorflow"]["memory_limits_mib"] = configured_limits

        if memory_growth is not None:
            if not isinstance(memory_growth, bool):
                raise TypeError("memory_growth must be a bool")
            if not physical_gpus:
                _log_warning("No physical GPUs detected; skipping memory growth toggle")
            for device in physical_gpus:
                if device.name in configured_limits:
                    _log_warning(
                        f"Skipping memory growth on {_tf_device_friendly_name(device)} because a memory limit is configured",
                    )
                    continue
                try:
                    tf.config.experimental.set_memory_growth(device, memory_growth)
                    summary.setdefault("tensorflow", {}).setdefault("memory_growth", {})[
                        device.name
                    ] = memory_growth
                    _log_success(
                        f"Set memory growth={'enabled' if memory_growth else 'disabled'} for {_tf_device_friendly_name(device)}",
                    )
                except RuntimeError as exc:
                    _log_error(f"Failed to set memory growth for {_tf_device_friendly_name(device)}: {exc}")

        # Enable deterministic ops if requested (TF versions that support it)
        if op_determinism is not None:
            if not isinstance(op_determinism, bool):
                raise TypeError("op_determinism must be a bool")
            if op_determinism:
                try:
                    # Available in TF 2.10+
                    tf.config.experimental.enable_op_determinism()
                    summary.setdefault("tensorflow", {})["op_determinism"] = True
                    _log_success("Enabled TensorFlow op determinism")
                except Exception as exc:
                    _log_error(f"Failed to enable op determinism: {exc}")
            else:
                # No TF API to explicitly disable; recommend env var via env_variables
                _log_warning(
                    "op_determinism=False requested, but disabling is not supported via TF API; "
                    "control via env_variables (e.g., TF_DETERMINISTIC_OPS=0) if needed",
                )

        # Configure global XLA JIT
        if xla_jit is not None:
            if not isinstance(xla_jit, bool):
                raise TypeError("xla_jit must be a bool")
            try:
                tf.config.optimizer.set_jit(xla_jit)
                summary.setdefault("tensorflow", {})["xla_jit"] = xla_jit
                _log_success(f"Set XLA JIT to {'enabled' if xla_jit else 'disabled'}")
            except Exception as exc:
                _log_error(f"Failed to set XLA JIT: {exc}")

        if intra_op_threads is not None:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
                summary["tensorflow"]["intra_op_threads"] = intra_op_threads
                _log_success(f"Set intra-op parallelism threads to {intra_op_threads}")
            except RuntimeError as exc:
                _log_error(f"Failed to set intra-op threads: {exc}")

        if inter_op_threads is not None:
            try:
                tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
                summary["tensorflow"]["inter_op_threads"] = inter_op_threads
                _log_success(f"Set inter-op parallelism threads to {inter_op_threads}")
            except RuntimeError as exc:
                _log_error(f"Failed to set inter-op threads: {exc}")

        if show_cuda_summary and verbosity > 0:
            # _log_info("TensorFlow GPU environment configured. Displaying CUDA summary:")
            try:
                print("\n")
                get_gpu_info()
            except Exception as exc:  # noqa: BLE001 - we want to surface any runtime issues
                _log_error(f"Failed to display GPU info: {exc}")

    finally:
        vp.verbose = previous_verbosity

    # return summary


def gpu_summary() -> None:
    """
    Prints a compact GPU summary similar to nvidia-smi output.
    """
    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    print(f"{timestamp}")
    print("+" + "-" * 88 + "+")
    print(
        f"| NVIDIA-SMI 470.xx                Driver Version: 470.xx       CUDA Version: {tf.sysconfig.get_build_info().get('cuda_version', 'N/A'):<4} |"
    )
    print("|" + "-" * 30 + "+" + "-" * 22 + "+" + "-" * 35 + "|")
    print("| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |")
    print("| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |")
    print("|                               |                      |               MIG M. |")
    print("|" + "=" * 88 + "|")

    gpu_data = _get_nvidia_smi_data()

    for gpu in gpu_data:
        name_short = gpu["name"][:16]
        used_gb = gpu["used_mb"] / 1024
        total_gb = gpu["total_mb"] / 1024
        temp = gpu["temperature"] if gpu["temperature"] != "N/A" else "--"
        util = gpu["utilization"] if gpu["utilization"] != "N/A" else "--"

        print(f"|   {gpu['index']}  {name_short:<16} Off  | 00000000:01:00.0 Off |                  N/A |")
        print(
            f"| N/A   {temp}C    P0    N/A /  N/A |  {used_gb:5.0f}MiB / {total_gb:5.0f}MiB |     {util}%      Default |"
        )
        print("|                               |                      |                  N/A |")
        print("+" + "-" * 88 + "+")

    print()
    print("+" + "-" * 88 + "+")
    print("| Processes:                                                                   |")
    print("|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |")
    print("|        ID   ID                                                   Usage      |")
    print("|" + "=" * 88 + "|")
    print("|  No running processes found                                                 |")
    print("+" + "-" * 88 + "+")


# ———————————————————————————————————————————————————————————————————————————— #
#                                Resources Utils                               #
# ———————————————————————————————————————————————————————————————————————————— #


def log_resources(log_dir: str, interval: int = 5, pid: Optional[int] = None, **kwargs) -> None:
    """Periodically record selected system metrics to CSV files.

    This helper spawns background threads that poll system resources using
    ``psutil`` and ``nvidia-smi``.  New entries are appended to CSV logs inside
    ``log_dir`` every ``interval`` seconds.  Because the logging threads run
    indefinitely, the resulting files can grow very large on long-running
    experiments.

    Args:
        log_dir (str): Directory where log files will be written.
        interval (int): Seconds between two consecutive samplings.
        pid (Optional[int]): Process ID whose CPU usage should also be logged. Defaults to the
            current process ID.
        **kwargs (Any): Flags indicating which resources should be logged. Supported
            flags are ``"cpu"``, ``"ram"``, ``"gpu"``, ``"cuda"`` and
            ``"tensorflow"``.

    Examples:
        >>> log_resources("logs", interval=10, pid=os.getpid(), cpu=True, ram=True, gpu=True)

    Notes:
        Ensure that ``log_dir`` has sufficient disk space available since the
        files are appended indefinitely.  CPU usage logged for ``pid`` represents
        the sum across all CPU cores and may exceed ``100``.
    """
    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    def log_cpu():
        """Log system and process CPU usage to a CSV file.

        Notes:
            The process CPU utilisation is aggregated across all cores and can
            therefore be greater than ``100`` on multi-core systems.
        """
        log_path = os.path.join(log_dir, "cpu_usage_log.csv")
        proc = psutil.Process(pid or os.getpid())
        proc.cpu_percent()
        with open(log_path, "w") as f:
            f.write("Timestamp,System_CPU_Usage(%),Process_CPU_Usage(%),Per-Core_Usage(%)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    system_cpu = psutil.cpu_percent()
                    process_cpu = proc.cpu_percent()
                    per_core_usage = psutil.cpu_percent(percpu=True)
                    f.write(f"{timestamp},{system_cpu},{process_cpu},{','.join(map(str, per_core_usage))}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_ram():
        """Log total, used and free system RAM to a CSV file.

        Warning:
            The output file grows without bound while the logger is running.
        """
        log_path = os.path.join(log_dir, "ram_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,Total(MB),Used(MB),Free(MB)\n")
            while True:
                try:
                    # Get memory statistics and convert from bytes to megabytes
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    mem = psutil.virtual_memory()
                    total = mem.total / (1024**2)
                    used = mem.used / (1024**2)
                    free = mem.available / (1024**2)

                    # Log the memory stats
                    f.write(f"{timestamp},{total:.2f},{used:.2f},{free:.2f}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_gpu():
        """Record GPU memory, utilisation and temperature statistics.

        Notes:
            This function relies on the ``nvidia-smi`` CLI tool being
            available in the system ``PATH``.

        Warning:
            The log file increases in size continuously while this thread is
            running.
        """
        log_path = os.path.join(log_dir, "gpu_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,GPU_ID,Memory_Used(MB),Memory_Total(MB),GPU_Utilization(%),Temperature(C)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # Execute nvidia-smi to get GPU stats
                    gpu_stats = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                    # Parse and write each GPU's data
                    for line in gpu_stats.split("\n"):
                        gpu_id, mem_used, mem_total, util, temp = map(int, line.split(","))
                        f.write(f"{timestamp},{gpu_id},{mem_used},{mem_total},{util},{temp}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_cuda():
        """Log CUDA memory usage by compute applications.

        Warning:
            The output file grows over time as new entries are appended.
        """
        log_path = os.path.join(log_dir, "cuda_usage_log.csv")
        with open(log_path, "w") as f:
            f.write("Timestamp,Process_Memory_Used(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # Query used CUDA memory by compute apps
                    cuda_mem_stats = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                    # Log memory usage or 0 MB if none used
                    if cuda_mem_stats:
                        f.write(f"{timestamp},{cuda_mem_stats} MB\n")
                    else:
                        f.write(f"{timestamp},0 MB\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    def log_tensorflow():
        """Log TensorFlow GPU memory usage to a CSV file.

        Warning:
            Continuous logging may produce very large files over time.
        """
        log_path = os.path.join(log_dir, "tensorflow_usage_log.csv")
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("Timestamp,Device,Memory_Allocated(MB),Memory_Peak(MB)\n")
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    gpus = tf.config.experimental.list_physical_devices("GPU")
                    for gpu in gpus:
                        device_name = gpu.name  # Extract device name
                        memory_info = tf.config.experimental.get_memory_info("GPU:0")

                        # Convert bytes to megabytes
                        allocated_memory = memory_info["current"] / (1024**2)
                        peak_memory = memory_info["peak"] / (1024**2)

                        # Write usage info to log
                        f.write(f"{timestamp},{device_name},{allocated_memory:.2f},{peak_memory:.2f}\n")
                    f.flush()
                    time.sleep(interval)
                except Exception as e:
                    f.write(f"Error: {e}\n")
                    break

    # Launch threads for selected logging targets
    if kwargs.get("cpu", False):
        Thread(target=log_cpu, daemon=True).start()
    if kwargs.get("ram", False):
        Thread(target=log_ram, daemon=True).start()
    if kwargs.get("gpu", True):
        Thread(target=log_gpu, daemon=True).start()
    if kwargs.get("cuda", True):
        Thread(target=log_cuda, daemon=True).start()
    if kwargs.get("tensorflow", True):
        Thread(target=log_tensorflow, daemon=True).start()


def _collect_cpu_usage() -> Dict[str, Any]:
    """Return current aggregate and per-core CPU usage percentages."""
    per_core_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    if per_core_percent:
        overall_percent = sum(per_core_percent) / len(per_core_percent)
    else:
        overall_percent = psutil.cpu_percent(interval=None)
        per_core_percent = []

    return {
        "metric": "cpu",
        "percent": overall_percent,
        "per_core_percent": per_core_percent,
    }


def _collect_ram_usage() -> Dict[str, Any]:
    """Return current RAM utilisation details in bytes and percent."""
    memory = psutil.virtual_memory()
    return {
        "metric": "ram",
        "total_bytes": memory.total,
        "used_bytes": memory.used,
        "available_bytes": memory.available,
        "percent": memory.percent,
    }


def _collect_disk_usage(path: str = os.sep) -> Dict[str, Any]:
    """Return disk usage statistics for the provided mount point."""
    usage = psutil.disk_usage(path)
    return {
        "metric": "disk",
        "path": path,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "percent": usage.percent,
    }


def _collect_gpu_memory() -> Dict[str, Any]:
    """Return GPU memory utilisation for devices reported by nvidia-smi."""
    gpu_data = _get_nvidia_smi_data()
    formatted_gpus = []
    for gpu in gpu_data:
        total_mb = gpu.get("total_mb")
        used_mb = gpu.get("used_mb")
        free_mb = gpu.get("free_mb")
        percent = (used_mb / total_mb * 100) if total_mb else None
        utilization_raw = gpu.get("utilization")
        utilization_percent = None
        if utilization_raw not in (None, "N/A"):
            try:
                utilization_percent = float(utilization_raw)
            except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                utilization_percent = None
        formatted_gpus.append(
            {
                "index": gpu.get("index"),
                "name": gpu.get("name"),
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "percent": percent,
                "utilization_percent": utilization_percent,
            }
        )

    return {"metric": "gpu_ram", "gpus": formatted_gpus}


def measure_current_system_resources(metrics: str = "cpu,ram,disk,gpu_ram") -> List[Dict[str, Any]]:
    """Collect system resource usage measurements for requested metrics.

    Args:
        metrics (str): Comma-separated list of metric identifiers to collect. Supported
            values are ``cpu``, ``ram``, ``disk``, ``gpu_ram`` and ``all``.
            Values are case-insensitive and surrounding whitespace is ignored.

    Returns:
        List[Dict[str, Any]]: One dictionary per requested metric. When a metric
        cannot be collected, the respective dictionary contains an ``error`` key
        with the failure reason.
    """

    metric_collectors: Dict[str, Callable[[], Dict[str, Any]]] = {
        "cpu": _collect_cpu_usage,
        "ram": _collect_ram_usage,
        "disk": _collect_disk_usage,
        "gpu_ram": _collect_gpu_memory,
    }

    normalized = {item.strip().lower() for item in metrics.split(",") if item.strip()}
    if not normalized:
        normalized = {"cpu", "ram", "disk", "gpu_ram"}
    if "all" in normalized:
        normalized = set(metric_collectors.keys())

    results: List[Dict[str, Any]] = []

    for metric_name in normalized:
        collector = metric_collectors.get(metric_name)
        if not collector:
            results.append({"metric": metric_name, "error": "unsupported metric"})
            continue

        try:
            results.append(collector())
        except Exception as exc:  # pragma: no cover - defensive safeguard
            vp.printf(f"Failed to collect {metric_name}: {exc}", tag="[ARARAS ERROR] ", color="red")
            results.append({"metric": metric_name, "error": str(exc)})

    return results
