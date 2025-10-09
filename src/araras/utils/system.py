import math
import os
import time
import statistics
import psutil
import subprocess
from datetime import datetime
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import tensorflow as tf

from araras.core import *
from araras.utils.misc import format_bytes, format_number_commas

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


def get_user_gpu_choice():
    """
    Prompts the user to select a GPU index and validates the input.

    Returns:
        str: Valid GPU index as string
    """
    available_gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(available_gpus)

    if num_gpus == 0:
        print(f"{RED}No GPUs available. Using CPU.{RESET}")
        return ""
    elif num_gpus == 1:
        # Get GPU data for single GPU
        gpu_data = _get_nvidia_smi_data()
        if gpu_data and len(gpu_data) > 0:
            gpu = gpu_data[0]
            free_gb = gpu["free_mb"] / 1024
            total_gb = gpu["total_mb"] / 1024
            print(
                f"Only one GPU available: {gpu['name']} ({GREEN}{free_gb:.1f}GB/{RED}{total_gb:.1f}GB free){RESET}"
            )
        else:
            print(f"Only one GPU available: {GREEN}{available_gpus[0].name}{RESET}")
        return "0"

    # Get GPU data for multiple GPUs
    gpu_data = _get_nvidia_smi_data()
    gpu_info_map = {gpu["index"]: gpu for gpu in gpu_data} if gpu_data else {}

    print(f"{BOLD}{BLUE}Available GPUs: {GREEN}{num_gpus}{RESET}")
    for i, gpu in enumerate(available_gpus):
        if i in gpu_info_map:
            gpu_info = gpu_info_map[i]
            free_gb = gpu_info["free_mb"] / 1024
            total_gb = gpu_info["total_mb"] / 1024
            print(f"  {CYAN}GPU {i}{RESET}: {gpu_info['name']} ({free_gb:.1f}GB/{total_gb:.1f}GB free)")
        else:
            print(f"  {CYAN}GPU {i}{RESET}: {gpu.name}")

    while True:
        try:
            user_input = input(f"\nEnter GPU index to use {YELLOW}(0-{num_gpus-1}): {RESET}").strip()
            gpu_index = int(user_input)

            if 0 <= gpu_index < num_gpus:
                logger.info(f"Selected GPU {gpu_index}")
                return str(gpu_index)
            else:
                print(f"{RED}Invalid index. Please enter a number between 0 and {num_gpus-1}.{RESET}")
        except ValueError:
            logger_error.error(f"{RED}Invalid input. Please enter a valid number.{RESET}")
        except KeyboardInterrupt:
            logger_error.warning(f"\n{RED}Operation cancelled. Using GPU 0 as default.{RESET}")


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
    print(f"{BOLD}TensorFlow Configuration{RESET}")
    print("=" * 80)
    print(f"Version        : {tf.__version__}")

    if tf.test.is_built_with_cuda():
        print(f"CUDA Support   : {GREEN}Yes{RESET}")
        build_info = tf.sysconfig.get_build_info()
        print(f"CUDA Version   : {build_info.get('cuda_version', 'Unknown')}")
        print(f"cuDNN Version  : {build_info.get('cudnn_version', 'Unknown')}")
    else:
        print(f"CUDA Support   : {RED}No (CPU only){RESET}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
       tf.test.gpu_device_name()


def _print_gpu_table(gpu_data):
    """Print GPU information in nvidia-smi style table format."""
    if not gpu_data:
        print(f"{RED}No GPU data available{RESET}")
        return

    print(f"\n{BOLD}GPU Information{RESET}")
    print("=" * 80)

    # Header
    header = f"{'GPU':<3} {'Name':<25} {'Memory Usage':<20} {'Temp':<6} {'Util':<6}"
    print(f"{BOLD}{header}{RESET}")
    print("-" * 80)

    # GPU rows
    for gpu in gpu_data:
        # Memory calculations
        total_gb = gpu["total_mb"] / 1024
        used_gb = gpu["used_mb"] / 1024
        utilization_pct = (used_gb / total_gb) * 100 if total_gb > 0 else 0

        # Format memory usage with color coding
        memory_str = f"{used_gb:6.1f}GB / {total_gb:6.1f}GB"
        if utilization_pct > 80:
            memory_color = RED
        elif utilization_pct > 50:
            memory_color = YELLOW
        else:
            memory_color = GREEN

        # Format temperature
        temp_str = f"{gpu['temperature']}C" if gpu["temperature"] != "N/A" else "N/A"

        # Format utilization
        util_str = f"{gpu['utilization']}%" if gpu["utilization"] != "N/A" else "N/A"

        # Print row
        row = (
            f"{gpu['index']:<3} "
            f"{gpu['name'][:24]:<25} "
            f"{memory_color}{memory_str:<20}{RESET} "
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
    print(f"\n{BOLD}{BLUE}TensorFlow GPU Monitor - {timestamp}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

    # TensorFlow configuration
    _print_tensorflow_info()

    # Get GPU data from nvidia-smi
    gpu_data = _get_nvidia_smi_data()

    # Print GPU table
    _print_gpu_table(gpu_data)

    # Print memory summary
    # _print_memory_summary(gpu_data)

    print(f"\n{BLUE}{'=' * 80}{RESET}")


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
            f.write(
                "Timestamp,System_CPU_Usage(%),Process_CPU_Usage(%),Per-Core_Usage(%)\n"
            )
            while True:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    system_cpu = psutil.cpu_percent()
                    process_cpu = proc.cpu_percent()
                    per_core_usage = psutil.cpu_percent(percpu=True)
                    f.write(
                        f"{timestamp},{system_cpu},{process_cpu},{','.join(map(str, per_core_usage))}\n"
                    )
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
            f.write(
                "Timestamp,GPU_ID,Memory_Used(MB),Memory_Total(MB),GPU_Utilization(%),Temperature(C)\n"
            )
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
                        f.write(
                            f"{timestamp},{gpu_id},{mem_used},{mem_total},{util},{temp}\n"
                        )
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
            logger_error.error(f"Failed to collect {metric_name}: {exc}")
            results.append({"metric": metric_name, "error": str(exc)})

    return results

