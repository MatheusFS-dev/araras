from typing import Any, Callable, Dict, Literal, Optional

import subprocess
import threading
import time
from pathlib import Path

import psutil
import pynvml

from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

Aggregation = Literal["delta", "peak"]


def logger_info(message: str) -> None:
    """Emit an informational log message using the global logger."""

    vp.printf(message, tag="[ARARAS INFO] ", color="blue")


def logger_warning(message: str) -> None:
    """Emit a warning log message using the global logger."""

    vp.printf(message, tag="[ARARAS WARNING] ", color="yellow")


def logger_error(message: str) -> None:
    """Emit an error log message using the global error logger."""

    vp.printf(message, tag="[ARARAS ERROR] ", color="red")


_UNAVAILABLE_WARNINGS: set[str] = set()
_NVML_INIT_LOCK = threading.Lock()
_NVML_INITIALIZED = False
_NVML_FAILURE: Optional[BaseException] = None
_NVML_HANDLES: Dict[int, Any] = {}

_RAPL_DISCOVERED = False
_RAPL_POWER_PATH: Optional[Path] = None
_RAPL_ENERGY_PATH: Optional[Path] = None


def _warn_once(metric_name: str, message: str) -> None:
    """Log ``message`` exactly once for ``metric_name``.

    Args:
        metric_name (str): Metric identifier associated with the warning.
        message (str): Text to log the first time the warning occurs.
    """

    if metric_name in _UNAVAILABLE_WARNINGS:
        return
    _UNAVAILABLE_WARNINGS.add(metric_name)
    logger_warning(message)


def _discover_rapl_paths() -> None:
    """Detect Intel RAPL files for CPU power estimation."""

    global _RAPL_DISCOVERED, _RAPL_POWER_PATH, _RAPL_ENERGY_PATH
    if _RAPL_DISCOVERED:
        return

    base_path = Path("/sys/class/powercap/intel-rapl")
    if not base_path.exists():
        _RAPL_DISCOVERED = True
        return

    for domain in sorted(base_path.glob("intel-rapl:*")):
        power_path = domain / "power_uw"
        energy_path = domain / "energy_uj"
        if power_path.exists():
            _RAPL_POWER_PATH = power_path
            break
        if energy_path.exists() and _RAPL_ENERGY_PATH is None:
            _RAPL_ENERGY_PATH = energy_path

    _RAPL_DISCOVERED = True


def _read_float_file(path: Path) -> Optional[float]:
    """Return the floating-point value stored in ``path``."""

    try:
        text = path.read_text().strip()
    except (OSError, ValueError):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _ensure_nvml() -> bool:
    """Initialise NVML once and report availability."""

    global _NVML_INITIALIZED, _NVML_FAILURE
    if _NVML_INITIALIZED:
        return True
    with _NVML_INIT_LOCK:
        if _NVML_INITIALIZED:
            return True
        try:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
            _NVML_FAILURE = None
        except Exception as exc:  # pragma: no cover - depends on system setup
            _NVML_FAILURE = exc
            _NVML_INITIALIZED = False
    return _NVML_INITIALIZED


def _get_nvml_handle(gpu_index: int) -> Optional[Any]:
    """Return the NVML handle for ``gpu_index`` if available."""

    if not _ensure_nvml():
        return None
    if gpu_index in _NVML_HANDLES:
        return _NVML_HANDLES[gpu_index]
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as exc:  # pragma: no cover - depends on runtime GPU setup
        _NVML_HANDLES[gpu_index] = None
        _warn_once(
            "gpu_metrics",
            "gpu metrics unavailable, requires NVIDIA GPU and NVML or nvidia-smi",
        )
        logger_error(f"NVML handle acquisition failed for GPU {gpu_index}: {exc}")
        return None
    _NVML_HANDLES[gpu_index] = handle
    return handle


def _nvidia_smi_query(field: str, gpu_index: int) -> Optional[float]:
    """Query ``field`` for ``gpu_index`` via ``nvidia-smi``.

    Args:
        field (str): Field understood by ``nvidia-smi --query-gpu``.
        gpu_index (int): GPU index to query.

    Returns:
        Optional[float]: Parsed floating-point result or ``None`` on failure.
    """

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index," + field,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        _warn_once(
            "gpu_metrics",
            "gpu metrics unavailable, requires NVIDIA GPU and NVML or nvidia-smi",
        )
        return None

    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in lines:
        try:
            index_str, value_str = line.split(",", 1)
        except ValueError:
            continue
        try:
            if int(index_str.strip()) != gpu_index:
                continue
        except ValueError:
            continue
        value = value_str.strip()
        if not value or value.lower() == "n/a":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _wait_interval(event: Optional[threading.Event], interval: float) -> bool:
    """Sleep for ``interval`` seconds using monotonic time.

    Args:
        event (Optional[threading.Event]): Optional cancellation event. When
            provided, the wait ends early if ``event`` is set.
        interval (float): Desired sleep duration in seconds.

    Returns:
        bool: ``True`` if the event was set during the wait, ``False`` otherwise.
    """

    if interval <= 0:
        if event is None:
            return False
        return event.is_set()

    deadline = time.monotonic() + interval
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False if event is None else event.is_set()
        sleep_time = min(remaining, 0.05)
        if event is None:
            time.sleep(sleep_time)
            continue
        if event.wait(sleep_time):
            return True


class ResourceMonitor:
    """Monitor resource usage for an arbitrary callable.

    The monitor samples configured metrics before a workload executes and while
    it is running.  Callers specify whether each metric should report the peak
    value during execution or the delta between peak values from the *during*
    and *before* phases.

    Args:
        metrics (Dict[str, Aggregation]): Mapping from metric names to
            aggregation policies. Use ``"delta"`` for peak differences and
            ``"peak"`` for raw peaks.
        before_repetitions (int): Number of baseline sampling passes before the
            workload starts. Defaults to ``1``.
        during_repetitions (int): Number of samples collected per loop
            iteration while the workload executes. Defaults to ``1``.
        sample_interval_s (float): Delay in seconds between samples. Defaults to
            ``0.1``.
        gpu_index (int): GPU index used by GPU-related metrics. Defaults to
            ``0``.
        verbose (bool): When ``True``, emit per-sample informational logs.

    Raises:
        ValueError: If ``metrics`` is empty, if ``before_repetitions`` or
            ``during_repetitions`` is less than ``1``.
    """

    def __init__(
        self,
        metrics: Dict[str, Aggregation],
        *,
        before_repetitions: int = 1,
        during_repetitions: int = 1,
        sample_interval_s: float = 0.1,
        gpu_index: int = 0,
        verbose: bool = True,
    ) -> None:
        if not metrics:
            raise ValueError("ResourceMonitor requires at least one metric")
        if before_repetitions < 1:
            raise ValueError("before_repetitions must be at least 1")
        if during_repetitions < 1:
            raise ValueError("during_repetitions must be at least 1")

        self.metrics = dict(metrics)
        self.before_repetitions = before_repetitions
        self.during_repetitions = during_repetitions
        self.sample_interval_s = sample_interval_s
        self.gpu_index = gpu_index
        self.verbose = verbose

        self.last_before_peaks: Dict[str, Optional[float]] = {}
        self.last_during_peaks: Dict[str, Optional[float]] = {}
        self.last_results: Dict[str, Optional[float]] = {}

        self._metric_getters = self._build_metric_getters()

    def _build_metric_getters(self) -> Dict[str, Callable[[], Optional[float]]]:
        """Create metric getter functions bound to the current configuration."""

        def cpu_util_percent() -> Optional[float]:
            try:
                return float(psutil.cpu_percent(interval=None))
            except Exception as exc:  # pragma: no cover - defensive safeguard
                logger_error(f"cpu_util_percent measurement failed: {exc}")
                return None

        def ram_used_bytes() -> Optional[float]:
            try:
                return float(psutil.virtual_memory().used)
            except Exception as exc:  # pragma: no cover - defensive safeguard
                logger_error(f"ram_used_bytes measurement failed: {exc}")
                return None

        def ram_util_percent() -> Optional[float]:
            try:
                return float(psutil.virtual_memory().percent)
            except Exception as exc:  # pragma: no cover - defensive safeguard
                logger_error(f"ram_util_percent measurement failed: {exc}")
                return None

        def cpu_power_rapl_w() -> Optional[float]:
            metric_name = "cpu_power_rapl_w"
            _discover_rapl_paths()
            if _RAPL_POWER_PATH is not None:
                value = _read_float_file(_RAPL_POWER_PATH)
                if value is None:
                    _warn_once(
                        metric_name,
                        "cpu_power_rapl_w unavailable, requires Intel RAPL, often sudo",
                    )
                    return None
                return value / 1_000_000.0
            if _RAPL_ENERGY_PATH is None:
                _warn_once(
                    metric_name,
                    "cpu_power_rapl_w unavailable, requires Intel RAPL, often sudo",
                )
                return None

            start = _read_float_file(_RAPL_ENERGY_PATH)
            if start is None:
                _warn_once(
                    metric_name,
                    "cpu_power_rapl_w unavailable, requires Intel RAPL, often sudo",
                )
                return None

            start_time = time.monotonic()
            _wait_interval(None, max(self.sample_interval_s, 0.05))
            end = _read_float_file(_RAPL_ENERGY_PATH)
            if end is None:
                return None
            elapsed = time.monotonic() - start_time
            if elapsed <= 0:
                return None
            delta_uj = end - start
            if delta_uj < 0:
                return None
            # Convert microjoules to joules and divide by elapsed time.
            return (delta_uj / 1_000_000.0) / elapsed

        def gpu_util_percent() -> Optional[float]:
            handle = _get_nvml_handle(self.gpu_index)
            if handle is not None:
                try:
                    return float(
                        pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    )
                except Exception:  # pragma: no cover - NVML dependent
                    handle = None
            value = _nvidia_smi_query("utilization.gpu", self.gpu_index)
            if value is None:
                _warn_once(
                    "gpu_metrics",
                    "gpu metrics unavailable, requires NVIDIA GPU and NVML or nvidia-smi",
                )
            return value

        def gpu_mem_used_bytes() -> Optional[float]:
            handle = _get_nvml_handle(self.gpu_index)
            if handle is not None:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    return float(info.used)
                except Exception:  # pragma: no cover - NVML dependent
                    handle = None
            value = _nvidia_smi_query("memory.used", self.gpu_index)
            if value is None:
                _warn_once(
                    "gpu_metrics",
                    "gpu metrics unavailable, requires NVIDIA GPU and NVML or nvidia-smi",
                )
                return None
            return value * 1024 * 1024

        def gpu_power_w() -> Optional[float]:
            handle = _get_nvml_handle(self.gpu_index)
            if handle is not None:
                try:
                    return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except Exception:  # pragma: no cover - NVML dependent
                    handle = None
            value = _nvidia_smi_query("power.draw", self.gpu_index)
            if value is None:
                _warn_once(
                    "gpu_metrics",
                    "gpu metrics unavailable, requires NVIDIA GPU and NVML or nvidia-smi",
                )
            return value

        metric_registry: Dict[str, Callable[[], Optional[float]]] = {
            "cpu_util_percent": cpu_util_percent,
            "cpu_power_rapl_w": cpu_power_rapl_w,
            "ram_used_bytes": ram_used_bytes,
            "ram_util_percent": ram_util_percent,
            "gpu_util_percent": gpu_util_percent,
            "gpu_mem_used_bytes": gpu_mem_used_bytes,
            "gpu_power_w": gpu_power_w,
        }

        getters: Dict[str, Callable[[], Optional[float]]] = {}
        for name in self.metrics:
            getter = metric_registry.get(name)
            if getter is None:
                raise ValueError(f"Unsupported metric '{name}' requested")
            getters[name] = getter
        return getters

    def _sample_metrics(self) -> Dict[str, Optional[float]]:
        """Sample all configured metrics once."""

        samples: Dict[str, Optional[float]] = {}
        for name, getter in self._metric_getters.items():
            try:
                samples[name] = getter()
            except Exception as exc:  # pragma: no cover - defensive safeguard
                logger_error(f"{name} measurement failed: {exc}")
                samples[name] = None
        return samples

    def _record_samples(
        self,
        store: Dict[str, Optional[float]],
        samples: Dict[str, Optional[float]],
        phase: str,
    ) -> None:
        for name, value in samples.items():
            if self.verbose:
                logger_info(f"measuring {name} {phase}, read={value}")
            if value is None:
                store[name] = None
                continue
            current = store.get(name)
            if current is None or value > current:
                store[name] = value

    def run_and_measure(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Optional[float]]:
        """Execute ``func`` while sampling metrics.

        Args:
            func (Callable[..., Any]): Target callable.
            *args (Any): Positional arguments forwarded to ``func``.
            **kwargs (Any): Keyword arguments forwarded to ``func``.

        Returns:
            Dict[str, Optional[float]]: Mapping from metric name to aggregated
                measurement according to the configured aggregation policy. A
                ``None`` value indicates the metric could not be measured.

        Raises:
            BaseException: Propagates any exception raised by ``func`` after a
                final sampling pass is completed.
        """

        before_peaks: Dict[str, Optional[float]] = {
            name: None for name in self.metrics
        }
        during_peaks: Dict[str, Optional[float]] = {
            name: None for name in self.metrics
        }

        for index in range(self.before_repetitions):
            samples = self._sample_metrics()
            self._record_samples(before_peaks, samples, "before")
            if index + 1 < self.before_repetitions:
                _wait_interval(None, self.sample_interval_s)

        stop_event = threading.Event()

        def sampler() -> None:
            while not stop_event.is_set():
                for _ in range(self.during_repetitions):
                    samples = self._sample_metrics()
                    self._record_samples(during_peaks, samples, "during")
                    if _wait_interval(stop_event, self.sample_interval_s):
                        return
                    if stop_event.is_set():
                        return

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()

        exc: Optional[BaseException] = None
        try:
            func(*args, **kwargs)
        except BaseException as error:  # pragma: no cover - propagates user code
            exc = error
        finally:
            stop_event.set()
            thread.join()
            final_samples = self._sample_metrics()
            self._record_samples(during_peaks, final_samples, "during")

        results: Dict[str, Optional[float]] = {}
        for name, aggregation in self.metrics.items():
            before_value = before_peaks.get(name)
            during_value = during_peaks.get(name)
            if before_value is None or during_value is None:
                results[name] = None
                continue
            if aggregation == "delta":
                results[name] = during_value - before_value
            else:
                results[name] = during_value

        self.last_before_peaks = before_peaks
        self.last_during_peaks = during_peaks
        self.last_results = results

        if exc is not None:
            raise exc

        return results
