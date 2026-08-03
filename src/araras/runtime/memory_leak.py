"""Conservative process-tree RSS trend detection for the runtime monitor."""

from typing import Callable, Deque, List, Optional, Tuple

import io
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass

import psutil

MIB_BYTES = 1024**2
SAMPLE_INTERVAL_SECONDS = 1.0
ANALYSIS_WINDOW_SECONDS = 600.0
EVALUATION_INTERVAL_SECONDS = 60.0
NET_GROWTH_THRESHOLD_MIB = 100.0
SLOPE_THRESHOLD_MIB_PER_MINUTE = 10.0
TREND_BLOCK_COUNT = 5
MIN_GROWING_BLOCK_TRANSITIONS = 3
MIN_BLOCK_GROWTH_MIB = 10.0


@dataclass(frozen=True)
class MemoryLeakEvidence:
    """Describe the RSS trend that triggered a possible-leak warning.

    Attributes:
        pid: Root PID of the monitored process tree.
        current_rss_mib: Most recent process-tree RSS in mebibytes.
        net_growth_mib: Difference between median RSS in the final and first
            minutes of the analysis window.
        slope_mib_per_minute: Least-squares RSS growth slope over the window.
        r_squared: Coefficient of determination for the fitted linear trend.
        warmup_seconds: Initial duration excluded from detection.
        window_seconds: Duration of the analyzed rolling window.
        graph_points: Per-minute median RSS points from process launch through
            detection, expressed as ``(elapsed_minutes, rss_mib)`` pairs.
    """

    pid: int
    current_rss_mib: float
    net_growth_mib: float
    slope_mib_per_minute: float
    r_squared: float
    warmup_seconds: float
    window_seconds: float
    graph_points: Tuple[Tuple[float, float], ...]


class ProcessMemoryLeakDetector:
    """Sample process-tree RSS and report one conservative growth signal."""

    def __init__(
        self,
        pid: int,
        warmup_seconds: float,
        warning_callback: Optional[Callable[[MemoryLeakEvidence], None]],
        sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        """Initialize a detector for one launched target PID.

        Args:
            pid (int): Positive root PID whose RSS and descendant RSS values
                are sampled.
            warmup_seconds (float): Positive finite initial duration excluded
                from trend analysis. Samples remain available for graphing.
            warning_callback (Optional[Callable[[MemoryLeakEvidence], None]]):
                Callback invoked once when all conservative leak criteria are
                met. It runs on the detector thread and must not mutate
                detector state. ``None`` disables trend analysis while keeping
                process-tree RSS collection active for lifecycle graphs.
            sample_interval_seconds (float): Positive finite delay between RSS
                samples in seconds. Defaults to one second. Smaller values add
                process-query overhead; larger values reduce trend resolution.

        Returns:
            None: The constructor stores configuration without starting a
            background thread.

        Raises:
            ValueError: If ``pid`` is not positive or either duration is not a
                positive finite number.
            TypeError: If ``warning_callback`` is neither callable nor
                ``None``.
        """
        if pid <= 0:
            raise ValueError("pid must be > 0")
        if not _is_positive_finite(warmup_seconds):
            raise ValueError("warmup_seconds must be a positive finite number")
        if not _is_positive_finite(sample_interval_seconds):
            raise ValueError("sample_interval_seconds must be a positive finite number")
        if warning_callback is not None and not callable(warning_callback):
            raise TypeError("warning_callback must be callable or None")

        self.pid = pid
        self.warmup_seconds = warmup_seconds
        self.warning_callback = warning_callback
        self.sample_interval_seconds = sample_interval_seconds
        self._analysis_samples: Deque[Tuple[float, float]] = deque()
        self._graph_points: List[Tuple[float, float]] = []
        self._bucket_index: Optional[int] = None
        self._bucket_values: List[float] = []
        self._next_evaluation_elapsed = warmup_seconds + ANALYSIS_WINDOW_SECONDS
        self._detected = False
        self._last_elapsed_seconds: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None

    def start(self) -> None:
        """Start background RSS sampling for the configured PID.

        Returns:
            None: A daemon sampling thread is created and started.

        Raises:
            RuntimeError: If this detector has already been started.
        """
        if self._thread is not None:
            raise RuntimeError("memory leak detector has already been started")

        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name=f"memory-leak-detector-{self.pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling and wait briefly for thread completion.

        Returns:
            None: The stop event is set and a live sampling thread is joined.

        Raises:
            RuntimeError: Not raised when the detector was never started or
                has already stopped.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 2.0))

    def add_sample(
        self,
        elapsed_seconds: float,
        rss_bytes: int,
        evaluate_immediately: bool = False,
    ) -> Optional[MemoryLeakEvidence]:
        """Add one RSS sample and evaluate the conservative trend when due.

        This method is public to keep the numerical detector independently
        testable without starting processes or sleeping.

        Args:
            elapsed_seconds (float): Non-negative finite seconds since this PID
                began monitoring. Values must be supplied in increasing order.
            rss_bytes (int): Non-negative process-tree resident memory in
                bytes.
            evaluate_immediately (bool): When ``False``, trend analysis follows
                the normal once-per-minute cadence. When ``True``, the new
                sample is evaluated immediately while still enforcing warm-up,
                complete-window, net-growth, slope, and sustained-growth
                requirements. Immediate evaluation performs no additional
                process query and has no effect after a warning. Defaults to
                ``False``.

        Returns:
            Optional[MemoryLeakEvidence]: Evidence when all leak thresholds are
            met for the first time, otherwise ``None``.

        Raises:
            ValueError: If elapsed time or RSS is invalid, elapsed time moves
                backwards, or a nominally complete window lacks samples in a
                required two-minute block.
        """
        if elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        if rss_bytes < 0:
            raise ValueError("rss_bytes must be >= 0")
        if self._last_elapsed_seconds is not None and elapsed_seconds < self._last_elapsed_seconds:
            raise ValueError("elapsed_seconds must not move backwards")
        self._last_elapsed_seconds = elapsed_seconds

        rss_mib = rss_bytes / float(MIB_BYTES)
        self._record_graph_value(elapsed_seconds, rss_mib)
        if (
            self.warning_callback is None
            or self._detected
            or elapsed_seconds < self.warmup_seconds
        ):
            return None

        # Keep only the raw samples needed for the current rolling trend. The
        # graph retains compact minute medians separately so its history still
        # begins at process launch.
        self._analysis_samples.append((elapsed_seconds, rss_mib))
        cutoff = elapsed_seconds - ANALYSIS_WINDOW_SECONDS
        while self._analysis_samples and self._analysis_samples[0][0] < cutoff:
            self._analysis_samples.popleft()

        # Trend fitting is intentionally limited to once per minute to avoid
        # reacting to short-lived allocation spikes and to minimize overhead.
        if not evaluate_immediately and elapsed_seconds < self._next_evaluation_elapsed:
            return None
        self._next_evaluation_elapsed = elapsed_seconds + EVALUATION_INTERVAL_SECONDS

        evidence = self._analyze_window(elapsed_seconds)
        if evidence is not None:
            self._detected = True
        return evidence

    def evaluate_current_sample(self) -> Optional[MemoryLeakEvidence]:
        """Sample live process-tree RSS and immediately evaluate the trend.

        This is used at a scheduled-restart boundary so the final eligible
        window is checked before the old PID is stopped.

        Returns:
            Optional[MemoryLeakEvidence]: Evidence when the final live sample
            satisfies every warning criterion, otherwise ``None``. Qualifying
            evidence is also delivered to the configured callback once.

        Raises:
            RuntimeError: If the detector was not started.
            psutil.NoSuchProcess: If the monitored root PID no longer exists.
            ValueError: If a nominally complete window lacks samples in a
                required two-minute block.
        """
        if self._started_at is None:
            raise RuntimeError("memory leak detector has not been started")

        rss_bytes = _get_process_tree_rss_bytes(self.pid)
        elapsed_seconds = time.monotonic() - self._started_at
        evidence = self.add_sample(
            elapsed_seconds,
            rss_bytes,
            evaluate_immediately=True,
        )
        if evidence is not None:
            if self.warning_callback is None:
                raise RuntimeError("memory leak evidence has no warning callback")
            self.warning_callback(evidence)
        return evidence

    def _sampling_loop(self) -> None:
        """Sample process-tree RSS until stopped, exited, or detected.

        Returns:
            None: Samples are forwarded to :meth:`add_sample`; qualifying
            evidence is passed to the configured callback exactly once.

        Raises:
            RuntimeError: If the detector thread runs without a start time.
        """
        if self._started_at is None:
            raise RuntimeError("memory leak detector start time is missing")

        while not self._stop_event.is_set():
            try:
                rss_bytes = _get_process_tree_rss_bytes(self.pid)
            except psutil.NoSuchProcess:
                return

            elapsed_seconds = time.monotonic() - self._started_at
            evidence = self.add_sample(elapsed_seconds, rss_bytes)
            if evidence is not None:
                if self.warning_callback is None:
                    raise RuntimeError("memory leak evidence has no warning callback")
                self.warning_callback(evidence)
            if self._stop_event.wait(self.sample_interval_seconds):
                return

    def get_graph_points(self) -> Tuple[Tuple[float, float], ...]:
        """Return compact process-tree RSS history collected so far.

        Call :meth:`stop` before this method when sampling runs on a background
        thread so the returned active-minute median is stable.

        Returns:
            Tuple[Tuple[float, float], ...]: Per-minute median points expressed
            as ``(elapsed_minutes, rss_mib)`` pairs, including the current
            partial minute when at least one sample exists.

        Raises:
            RuntimeError: If no RSS sample has been collected.
        """
        if self._last_elapsed_seconds is None:
            raise RuntimeError("no process memory samples were collected")

        graph_points = list(self._graph_points)
        if self._bucket_values:
            graph_points.append(
                (
                    self._last_elapsed_seconds / 60.0,
                    float(statistics.median(self._bucket_values)),
                )
            )
        return tuple(graph_points)

    def _record_graph_value(self, elapsed_seconds: float, rss_mib: float) -> None:
        """Accumulate compact per-minute medians for warning graph creation.

        Args:
            elapsed_seconds (float): Seconds since this PID began monitoring.
            rss_mib (float): Process-tree RSS in mebibytes for this sample.

        Returns:
            None: The active minute bucket or completed graph history is
            updated in place.

        Raises:
            RuntimeError: Not raised for valid samples supplied by
                :meth:`add_sample`.
        """
        bucket_index = int(elapsed_seconds // 60.0)
        if self._bucket_index is None:
            self._bucket_index = bucket_index
        elif bucket_index != self._bucket_index:
            self._graph_points.append(
                (
                    (self._bucket_index + 0.5),
                    float(statistics.median(self._bucket_values)),
                )
            )
            self._bucket_index = bucket_index
            self._bucket_values = []
        self._bucket_values.append(rss_mib)

    def _analyze_window(self, elapsed_seconds: float) -> Optional[MemoryLeakEvidence]:
        """Return evidence when the current rolling window meets all criteria.

        Args:
            elapsed_seconds (float): Elapsed duration of the newest sample in
                seconds.

        Returns:
            Optional[MemoryLeakEvidence]: Quantified growth evidence, or
            ``None`` when the window is incomplete or any threshold fails.

        Raises:
            ValueError: If a nominally complete window lacks samples in a
                required two-minute block.
        """
        samples = list(self._analysis_samples)
        if len(samples) < 2:
            return None
        if samples[-1][0] - samples[0][0] < ANALYSIS_WINDOW_SECONDS - self.sample_interval_seconds:
            return None

        # Endpoint medians make the net-growth gate resistant to isolated RSS
        # spikes. The later slope and block gates require speed and persistence.
        first_minute_end = samples[0][0] + 60.0
        last_minute_start = samples[-1][0] - 60.0
        first_values = [rss for elapsed, rss in samples if elapsed <= first_minute_end]
        last_values = [rss for elapsed, rss in samples if elapsed >= last_minute_start]
        if not first_values or not last_values:
            return None

        net_growth_mib = float(statistics.median(last_values)) - float(
            statistics.median(first_values)
        )
        if net_growth_mib < NET_GROWTH_THRESHOLD_MIB:
            return None

        slope, r_squared = _fit_linear_trend(samples)
        if slope < SLOPE_THRESHOLD_MIB_PER_MINUTE:
            return None

        # A leak may grow in allocation steps rather than along a straight
        # line. Requiring growth across most two-minute block transitions
        # rejects one-time jumps without rejecting sustained staircase growth.
        block_medians = _calculate_block_medians(
            samples,
            window_end_seconds=elapsed_seconds,
        )
        growing_transitions = sum(
            next_median - current_median >= MIN_BLOCK_GROWTH_MIB
            for current_median, next_median in zip(
                block_medians,
                block_medians[1:],
            )
        )
        if growing_transitions < MIN_GROWING_BLOCK_TRANSITIONS:
            return None

        return MemoryLeakEvidence(
            pid=self.pid,
            current_rss_mib=samples[-1][1],
            net_growth_mib=net_growth_mib,
            slope_mib_per_minute=slope,
            r_squared=r_squared,
            warmup_seconds=self.warmup_seconds,
            window_seconds=ANALYSIS_WINDOW_SECONDS,
            graph_points=self.get_graph_points(),
        )


def _calculate_block_medians(
    samples: List[Tuple[float, float]],
    window_end_seconds: float,
) -> Tuple[float, ...]:
    """Calculate five consecutive two-minute RSS medians.

    Args:
        samples (List[Tuple[float, float]]): Ordered ``(elapsed_seconds,
            rss_mib)`` samples covering the configured analysis window.
        window_end_seconds (float): Elapsed time in seconds at the end of the
            rolling analysis window.

    Returns:
        Tuple[float, ...]: One median RSS value in MiB for each of the five
        consecutive two-minute blocks.

    Raises:
        ValueError: If any block has no samples.

    Examples:
        Ten minutes of one-second samples returns five median values.
    """
    window_start_seconds = window_end_seconds - ANALYSIS_WINDOW_SECONDS
    block_seconds = ANALYSIS_WINDOW_SECONDS / TREND_BLOCK_COUNT
    block_values: List[List[float]] = [[] for _ in range(TREND_BLOCK_COUNT)]

    for elapsed_seconds, rss_mib in samples:
        block_index = int((elapsed_seconds - window_start_seconds) // block_seconds)
        if block_index == TREND_BLOCK_COUNT:
            block_index -= 1
        if 0 <= block_index < TREND_BLOCK_COUNT:
            block_values[block_index].append(rss_mib)

    if any(not values for values in block_values):
        raise ValueError("memory trend window has an empty two-minute block")
    return tuple(float(statistics.median(values)) for values in block_values)


def render_memory_leak_graph(evidence: MemoryLeakEvidence) -> bytes:
    """Render an inline PNG showing RAM growth through leak detection.

    Args:
        evidence (MemoryLeakEvidence): Qualified leak evidence containing the
            compact graph series and warm-up duration.

    Returns:
        bytes: PNG image bytes suitable for a MIME inline image.

    Raises:
        ValueError: If the evidence has no graph points.
        ImportError: If Matplotlib is unavailable.
        OSError: If Matplotlib cannot encode the in-memory PNG.
        RuntimeError: If Matplotlib cannot initialize or render the figure.

    Examples:
        ``render_memory_leak_graph(evidence)`` returns bytes beginning with the
        PNG signature ``b"\\x89PNG"``.
    """
    return _render_memory_usage_graph(
        evidence.graph_points,
        title="Possible Memory Leak: Process-Tree RAM Over Time",
        marker_label="Warning",
        warmup_minutes=evidence.warmup_seconds / 60.0,
    )


def render_scheduled_restart_graph(
    graph_points: Tuple[Tuple[float, float], ...],
) -> bytes:
    """Render process-tree RAM history for a successful scheduled restart.

    Args:
        graph_points (Tuple[Tuple[float, float], ...]): Non-empty per-minute
            ``(elapsed_minutes, rss_mib)`` points captured for the stopped PID.

    Returns:
        bytes: PNG image bytes suitable for an inline MIME image.

    Raises:
        ValueError: If no graph points are provided.
        ImportError: If Matplotlib is unavailable.
        OSError: If Matplotlib cannot encode the in-memory PNG.
        RuntimeError: If Matplotlib cannot initialize or render the figure.

    Examples:
        ``render_scheduled_restart_graph(((0.0, 512.0),))`` returns PNG bytes.
    """
    return _render_memory_usage_graph(
        graph_points,
        title="Scheduled Restart: Process-Tree RAM Usage",
        marker_label="Scheduled restart",
        warmup_minutes=None,
    )


def _render_memory_usage_graph(
    graph_points: Tuple[Tuple[float, float], ...],
    title: str,
    marker_label: str,
    warmup_minutes: Optional[float],
) -> bytes:
    """Render compact process-tree RSS points as an in-memory PNG.

    Args:
        graph_points (Tuple[Tuple[float, float], ...]): Non-empty per-minute
            ``(elapsed_minutes, rss_mib)`` points.
        title (str): Figure title describing the lifecycle event.
        marker_label (str): Legend label for the final elapsed-time marker.
        warmup_minutes (Optional[float]): Non-negative warm-up duration to
            shade, or ``None`` when no warm-up region should be shown.

    Returns:
        bytes: Encoded PNG image bytes.

    Raises:
        ValueError: If graph points are empty or warm-up is negative.
        ImportError: If Matplotlib is unavailable.
        OSError: If Matplotlib cannot encode the in-memory PNG.
        RuntimeError: If Matplotlib cannot initialize or render the figure.
    """
    if not graph_points:
        raise ValueError("memory usage graph requires at least one point")
    if warmup_minutes is not None and warmup_minutes < 0:
        raise ValueError("warmup_minutes must be non-negative or None")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed_minutes = [point[0] for point in graph_points]
    rss_mib = [point[1] for point in graph_points]
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(elapsed_minutes, rss_mib, color="#000000", linewidth=1.8, label="RSS")
    if warmup_minutes is not None:
        axis.axvspan(0.0, warmup_minutes, color="#d7dbe2", alpha=0.6, label="Warm-up")
    axis.axvline(
        elapsed_minutes[-1],
        color="#b54708",
        linestyle="--",
        label=marker_label,
    )
    axis.set_title(title)
    axis.set_xlabel("Elapsed Time (minutes)")
    axis.set_ylabel("Monitored Script RAM (MiB)")
    axis.grid(True, color="#d7dbe2", linewidth=0.6)
    axis.legend(loc="best")
    figure.tight_layout()

    output = io.BytesIO()
    try:
        figure.savefig(output, format="png")
        return output.getvalue()
    finally:
        plt.close(figure)


def _fit_linear_trend(samples: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Fit RSS against elapsed minutes and return slope and R-squared.

    Args:
        samples (List[Tuple[float, float]]): At least two ordered
            ``(elapsed_seconds, rss_mib)`` samples with nonzero time span.

    Returns:
        Tuple[float, float]: Slope in MiB per minute and coefficient of
        determination in the range normally bounded by zero and one.

    Raises:
        ValueError: If fewer than two samples are supplied, elapsed time has no
            variance, or RSS has no variance.
    """
    if len(samples) < 2:
        raise ValueError("at least two samples are required")

    start_time = samples[0][0]
    x_values = [(elapsed - start_time) / 60.0 for elapsed, _ in samples]
    y_values = [rss for _, rss in samples]
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    x_variance = sum((value - mean_x) ** 2 for value in x_values)
    y_variance = sum((value - mean_y) ** 2 for value in y_values)
    if x_variance == 0:
        raise ValueError("sample elapsed times have no variance")
    if y_variance == 0:
        raise ValueError("sample RSS values have no variance")

    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    ) / x_variance
    intercept = mean_y - slope * mean_x
    residual_sum = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x_values, y_values)
    )
    r_squared = 1.0 - (residual_sum / y_variance)
    return slope, r_squared


def _get_process_tree_rss_bytes(pid: int) -> int:
    """Return combined RSS for a live root PID and its descendants.

    Args:
        pid (int): Positive root process ID.

    Returns:
        int: Combined resident memory in bytes for processes still alive while
        the sample is collected.

    Raises:
        psutil.NoSuchProcess: If the root process no longer exists.
    """
    root_process = psutil.Process(pid)
    processes = [root_process] + root_process.children(recursive=True)
    total_rss = 0
    for process in processes:
        try:
            total_rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            # Descendants may exit while the process tree is being sampled.
            if process.pid == pid:
                raise
    return total_rss


def _is_positive_finite(value: float) -> bool:
    """Return whether a numeric duration is finite and strictly positive.

    Args:
        value (float): Duration or interval to inspect.

    Returns:
        bool: ``True`` only for finite values greater than zero.

    Raises:
        TypeError: If ``value`` does not support numeric comparison.
    """
    return value > 0 and math.isfinite(value)
